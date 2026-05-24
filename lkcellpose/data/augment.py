import numpy as np
import cv2
import random


class CellposeAugment:
    """Cellpose-SAM style augmentations matching cellpose v4.

    Key differences from previous implementation:
      - Arbitrary rotation (0-2π) instead of 90° only
      - Combined rotation + scaling via cv2.warpAffine (single affine)
      - Flow vectors properly rotated by the same angle
      - Contrast range clamped to positive values

    Matches cellpose/transforms.py:random_rotate_and_resize
    """

    def __init__(self, scale_range=0.5, grayscale_prob=0.1, invert_prob=0.25,
                 channel_dropout_prob=0.1, brightness_std=0.2, contrast_range=(0.5, 1.5),
                 degradation_prob=0.5, hflip=True, vflip=True, rotation=True,
                 target_size=256):
        self.scale_range = scale_range
        self.grayscale_prob = grayscale_prob
        self.invert_prob = invert_prob
        self.channel_dropout_prob = channel_dropout_prob
        self.brightness_std = brightness_std
        self.contrast_range = contrast_range
        self.degradation_prob = degradation_prob
        self.hflip = hflip
        self.vflip = vflip
        self.rotation = rotation
        self.target_size = target_size

    def __call__(self, sample: dict) -> dict:
        """Augment a sample dict with keys: img, flows, cellprob, class_map.
        Optional keys: labels (integer instance map).
        """
        img = sample["img"].copy()
        has_flows = "flows" in sample

        # ---- Step 1: Affine transform (rotation + scale) matching cellpose ----
        # cellpose uses random_rotate_and_resize which combines rotation + scale
        # into a single cv2.warpAffine, then rotates flow vectors afterward.
        if self.rotation or self.scale_range > 0:
            theta, scale, M = self._random_affine(img, has_flows=has_flows)
            img, sample = self._apply_affine(img, sample, M, theta, scale)

        # ---- Step 2: Horizontal flip ----
        if self.hflip and random.random() < 0.5:
            if img.ndim == 3:
                img = img[:, ::-1].copy()
            else:
                img = img[::-1].copy()
            if has_flows:
                sample["flows"] = sample["flows"][:, :, ::-1].copy()
                sample["flows"][1] = -sample["flows"][1]  # flip X-flow
            for key in ["class_map", "cellprob", "labels"]:
                if key in sample:
                    arr = sample[key]
                    sample[key] = arr[:, ::-1].copy() if arr.ndim == 3 else arr[::-1].copy()

        # ---- Step 3: Vertical flip ----
        if self.vflip and random.random() < 0.5:
            if img.ndim == 3:
                img = img[::-1].copy()
            else:
                img = img[::-1].copy()
            if has_flows:
                sample["flows"] = sample["flows"][:, ::-1].copy()
                sample["flows"][0] = -sample["flows"][0]  # flip Y-flow
            for key in ["class_map", "cellprob", "labels"]:
                if key in sample:
                    arr = sample[key]
                    sample[key] = arr[::-1, :].copy() if arr.ndim == 3 else arr[::-1].copy()

        # ---- Step 4: Photometric augmentations ----
        if random.random() < self.grayscale_prob and img.ndim == 3:
            img = self._to_grayscale(img)

        if random.random() < self.invert_prob:
            img = 1.0 - img

        if img.ndim == 3 and random.random() < self.channel_dropout_prob:
            c = random.randint(0, img.shape[2] - 1)
            img[:, :, c] = img[:, :, random.randint(0, img.shape[2] - 1)]

        if self.brightness_std > 0:
            for c in range(img.shape[2] if img.ndim == 3 else 1):
                delta = random.gauss(0, self.brightness_std)
                if img.ndim == 3:
                    img[:, :, c] = np.clip(img[:, :, c] + delta, 0, 1)
                else:
                    img = np.clip(img + delta, 0, 1)

        # Contrast: only positive factors to avoid inverting or zeroing image
        contrast_factor = random.uniform(*self.contrast_range)
        if abs(contrast_factor - 1.0) > 0.01:
            img = np.clip(img * contrast_factor, 0, 1)

        sample["img"] = img.astype(np.float32)
        return sample

    def _random_affine(self, img, has_flows=False):
        """Generate random rotation + scale affine transform.

        Matches cellpose/transforms.py:random_rotate_and_resize logic.
        Returns (theta, scale, M) where M is the 2x3 affine matrix for cv2.warpAffine.
        """
        h, w = img.shape[:2]
        xy = self.target_size

        theta = random.random() * np.pi * 2 if self.rotation else 0.0

        if self.scale_range is not None:
            scale = (1 - self.scale_range / 2) + self.scale_range * random.random()
        else:
            scale = 2 ** (4 * random.random() - 2)

        # Translation offset (random crop within scaled image)
        dxy = np.maximum(0, np.array([w * scale - xy, h * scale - xy]))
        dxy = (np.random.rand(2) - 0.5) * dxy

        # Build affine: center, center+[1,0], center+[0,1] → transformed
        cc = np.array([w / 2, h / 2])
        cc1 = cc - np.array([w - xy, h - xy]) / 2 + dxy
        pts1 = np.float32([cc, cc + np.array([1, 0]), cc + np.array([0, 1])])
        pts2 = np.float32([
            cc1,
            cc1 + scale * np.array([np.cos(theta), np.sin(theta)]),
            cc1 + scale * np.array([np.cos(np.pi / 2 + theta), np.sin(np.pi / 2 + theta)])
        ])
        M = cv2.getAffineTransform(pts1, pts2)
        return theta, scale, M

    def _apply_affine(self, img, sample, M, theta, scale):
        """Apply affine transform to image and all spatial arrays in sample.

        Matches cellpose's approach:
          - Images: bilinear interpolation
          - Labels/class_map/cellprob: nearest-neighbor
          - Flows: bilinear interpolation, then vector rotation by theta
        """
        xy = self.target_size

        # Transform image
        if img.ndim == 3:
            nchan = img.shape[2]
            out_img = np.zeros((xy, xy, nchan), dtype=np.float32)
            for c in range(nchan):
                out_img[:, :, c] = cv2.warpAffine(img[:, :, c], M, (xy, xy),
                                                  flags=cv2.INTER_LINEAR)
        else:
            out_img = cv2.warpAffine(img, M, (xy, xy), flags=cv2.INTER_LINEAR)
        img = out_img

        # Transform labels, class_map, cellprob (nearest-neighbor)
        for key in ["labels", "class_map", "cellprob"]:
            if key in sample:
                arr = sample[key]
                if arr.ndim == 3:
                    out = np.zeros((arr.shape[0], xy, xy), dtype=arr.dtype)
                    for c in range(arr.shape[0]):
                        out[c] = cv2.warpAffine(arr[c], M, (xy, xy),
                                                flags=cv2.INTER_NEAREST)
                else:
                    out = cv2.warpAffine(arr, M, (xy, xy), flags=cv2.INTER_NEAREST)
                sample[key] = out

        # Transform flows (bilinear), then rotate flow vectors
        if "flows" in sample:
            flows = sample["flows"]
            out_flows = np.zeros((flows.shape[0], xy, xy), dtype=np.float32)
            for c in range(flows.shape[0]):
                out_flows[c] = cv2.warpAffine(flows[c], M, (xy, xy),
                                              flags=cv2.INTER_LINEAR)

            # Rotate flow vectors by theta (matching cellpose/transforms.py:1080-1084)
            # Cellpose:
            #   lbl[n, -2] = (-v1 * sin(-theta) + v2 * cos(-theta))   → new Y-flow
            #   lbl[n, -1] = ( v1 * cos(-theta) + v2 * sin(-theta))   → new X-flow
            # where v1 = X-flow, v2 = Y-flow
            if self.rotation and abs(theta) > 1e-8:
                x_flow = out_flows[1].copy()  # v1 = X-flow
                y_flow = out_flows[0].copy()  # v2 = Y-flow
                out_flows[0] = -x_flow * np.sin(-theta) + y_flow * np.cos(-theta)
                out_flows[1] =  x_flow * np.cos(-theta) + y_flow * np.sin(-theta)

            sample["flows"] = out_flows

        return img, sample

    def _to_grayscale(self, img):
        if img.shape[2] >= 2:
            gray = img.mean(axis=2, keepdims=True)
        else:
            gray = img
        return np.repeat(gray, 3, axis=2)
