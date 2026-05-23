import numpy as np
import random


class CellposeAugment:
    """Cellpose-SAM style augmentations for training."""

    def __init__(self, scale_range=(0.25, 4.0), grayscale_prob=0.1, invert_prob=0.25,
                 channel_dropout_prob=0.1, brightness_std=0.2, contrast_range=(-2.0, 2.0),
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
        spatial_keys_2d = ["class_map", "cellprob", "labels"]
        spatial_keys_3d = ["flows"]

        if self.rotation:
            k = random.randint(0, 3)
            img = np.rot90(img, k)
            for key in spatial_keys_3d:
                if key in sample:
                    sample[key] = np.rot90(sample[key], k, axes=(1, 2))
            for key in spatial_keys_2d:
                if key in sample:
                    sample[key] = np.rot90(sample[key], k, axes=(0, 1))

        if self.hflip and random.random() < 0.5:
            if img.ndim == 3:
                img = img[:, ::-1].copy()
            else:
                img = img[::-1].copy()
            if "flows" in sample:
                sample["flows"] = sample["flows"][:, :, ::-1].copy()
                sample["flows"][1] = -sample["flows"][1]
            for key in spatial_keys_2d:
                if key in sample:
                    arr = sample[key]
                    sample[key] = arr[:, ::-1].copy() if arr.ndim == 3 else arr[::-1].copy()

        if self.vflip and random.random() < 0.5:
            if img.ndim == 3:
                img = img[::-1].copy()
            else:
                img = img[::-1].copy()
            if "flows" in sample:
                sample["flows"] = sample["flows"][:, ::-1].copy()
                sample["flows"][0] = -sample["flows"][0]
            for key in spatial_keys_2d:
                if key in sample:
                    arr = sample[key]
                    sample[key] = arr[::-1, :].copy() if arr.ndim == 3 else arr[::-1].copy()

        if self.scale_range[0] != 1.0 or self.scale_range[1] != 1.0:
            log_scale = random.uniform(
                np.log(self.scale_range[0]), np.log(self.scale_range[1])
            )
            scale = np.exp(log_scale)
            if abs(scale - 1.0) > 0.01:
                from scipy.ndimage import zoom
                if img.ndim == 3:
                    img = zoom(img, (scale, scale, 1.0), order=1)
                else:
                    img = zoom(img, (scale, scale), order=1)
                if "flows" in sample:
                    sample["flows"] = zoom(sample["flows"], (1, scale, scale), order=1)
                for key in spatial_keys_2d:
                    if key in sample:
                        sample[key] = zoom(sample[key].astype(np.float32),
                                           (scale, scale), order=0).astype(sample[key].dtype)
                img, sample = self._random_crop(img, sample)

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

        contrast_factor = random.uniform(*self.contrast_range)
        if abs(contrast_factor) > 0.01:
            img = np.clip(img * contrast_factor, 0, 1)

        sample["img"] = img.astype(np.float32)
        return sample

    def _random_crop(self, img, sample):
        """Crop/pad to target_size x target_size."""
        ts = self.target_size
        h, w = img.shape[:2]
        spatial_keys_2d = ["class_map", "cellprob", "labels"]
        spatial_keys_3d = ["flows"]

        if h >= ts and w >= ts:
            top = random.randint(0, h - ts)
            left = random.randint(0, w - ts)
            if img.ndim == 3:
                img = img[top:top + ts, left:left + ts, :]
            else:
                img = img[top:top + ts, left:left + ts]
            for key in spatial_keys_3d:
                if key in sample:
                    sample[key] = sample[key][:, top:top + ts, left:left + ts]
            for key in spatial_keys_2d:
                if key in sample:
                    sample[key] = sample[key][top:top + ts, left:left + ts]
        else:
            pad_h = max(0, ts - h)
            pad_w = max(0, ts - w)
            if img.ndim == 3:
                img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            else:
                img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
            for key in spatial_keys_3d:
                if key in sample:
                    sample[key] = np.pad(sample[key], ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
            for key in spatial_keys_2d:
                if key in sample:
                    sample[key] = np.pad(sample[key], ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        return img, sample

    def _to_grayscale(self, img):
        if img.shape[2] >= 2:
            gray = img.mean(axis=2, keepdims=True)
        else:
            gray = img
        return np.repeat(gray, 3, axis=2)
