import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from lkcellpose.cfg import get_cfg, DEFAULT_CFG_PATH
from lkcellpose.utils import LOGGER
from lkcellpose.utils.torch_utils import select_device, smart_inference_mode
from lkcellpose.data.transforms import normalize_img
from lkcellpose.engine.results import Results


class BasePredictor:
    def __init__(self, cfg=DEFAULT_CFG_PATH, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        overrides.setdefault("mode", "predict")
        self.args = get_cfg(cfg, overrides)
        self.device = select_device(self.args.get("device", "auto"))
        self.model = None
        self.callbacks = _callbacks or {}
        self.tile_overlap = self.args.get("tile_overlap", 0.1)
        self.cellprob_threshold = self.args.get("cellprob_threshold", 0.0)
        self.flow_threshold = self.args.get("flow_threshold", 0.4)
        self.min_size = self.args.get("min_size", 15)
        self.niter = self.args.get("niter", 200)
        self.input_size = self.args.get("input_size", 256)

    @smart_inference_mode()
    def __call__(self, source, model=None, **kwargs):
        if model is not None:
            self.model = model.to(self.device)
            self.model.eval()
        if isinstance(source, (str, Path)):
            source = self._load_image(source)
        if isinstance(source, np.ndarray):
            source = [source]
        results = []
        for img in tqdm(source, desc="Predicting"):
            result = self.predict_single(img)
            results.append(result)
        return results

    def predict_single(self, img):
        img_np = normalize_img(img) if img.max() > 1 else img.astype(np.float32)
        h, w = img_np.shape[:2]
        if h != self.input_size or w != self.input_size:
            tiles, positions = self._make_tiles(img_np)
            pred_tiles = []
            for tile in tiles:
                tile_t = torch.from_numpy(tile).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                    pred = self.model(tile_t)
                pred_tiles.append(pred[0].cpu().numpy())
            pred_full = self._average_tiles(pred_tiles, positions, (pred_tiles[0].shape[0], h, w))
        else:
            img_t = torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                pred = self.model(img_t)
            pred_full = pred[0].cpu().numpy()

        masks, classes = self._postprocess(pred_full)
        return Results(orig_img=img, masks=masks, classes=classes, pred=pred_full)

    def _make_tiles(self, img):
        ts = self.input_size
        overlap = int(ts * self.tile_overlap)
        stride = ts - overlap
        h, w = img.shape[:2]
        tiles = []
        positions = []
        for y in range(0, max(h - ts + 1, 1), stride):
            for x in range(0, max(w - ts + 1, 1), stride):
                y_end = min(y + ts, h)
                x_end = min(x + ts, w)
                y_start = y_end - ts if y_end - ts >= 0 else 0
                x_start = x_end - ts if x_end - ts >= 0 else 0
                tile = img[y_start:y_end, x_start:x_end]
                if tile.shape[0] < ts or tile.shape[1] < ts:
                    padded = np.zeros((ts, ts, img.shape[2]), dtype=img.dtype)
                    padded[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded
                tiles.append(tile)
                positions.append((y_start, x_start, y_start + ts, x_start + ts))
        return tiles, positions

    def _average_tiles(self, pred_tiles, positions, out_shape):
        nch, h, w = out_shape
        output = np.zeros((nch, h, w), dtype=np.float32)
        count = np.zeros((h, w), dtype=np.float32)
        ts = self.input_size
        for pred, (y1, x1, y2, x2) in zip(pred_tiles, positions):
            ph, pw = min(ts, h - y1), min(ts, w - x1)
            output[:, y1:y1+ph, x1:x1+pw] += pred[:, :ph, :pw]
            count[y1:y1+ph, x1:x1+pw] += 1.0
        count = np.maximum(count, 1.0)
        output /= count[np.newaxis]
        return output

    def _postprocess(self, pred):
        from cellpose import dynamics
        flow_y = pred[0]
        flow_x = pred[1]
        cellprob = pred[2]

        flow = np.stack([flow_y, flow_x], axis=0)
        cellprob_mask = 1.0 / (1.0 + np.exp(-cellprob)) > self.cellprob_threshold

        labels = dynamics.compute_masks(
            flow, cellprob_mask,
            flow_threshold=self.flow_threshold,
            min_size=self.min_size,
            niter=self.niter,
            device=self.device,
        )
        if isinstance(labels, tuple):
            labels = labels[0]

        classes = None
        if pred.shape[0] > 3:
            class_logits = pred[3:]
            class_pred = np.argmax(class_logits, axis=0)
            classes = class_pred

        return labels, classes

    def _load_image(self, path):
        from PIL import Image
        img = np.array(Image.open(path).convert("RGB"))
        return img
