"""MegaLoc backbone wrapper for feature extraction."""

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms


class MegaLocBackbone:
    """
    Wrapper around the MegaLoc VPR model.

    Loads via torch.hub, preprocesses images (resize to 322x322,
    ImageNet normalize), and extracts 8448-dim L2-normalized descriptors.
    """

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self.model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
        self.model = self.model.to(self.device).eval()
        self.input_size = 322
        self.descriptor_dim = 8448
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.input_size, self.input_size), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Convert BGR/RGB numpy image to preprocessed tensor."""
        if image.shape[2] == 3 and image.dtype == np.uint8:
            # OpenCV loads BGR, convert to RGB
            image = image[:, :, ::-1].copy()
        return self.transform(image)

    @torch.no_grad()
    def extract(self, images: list[np.ndarray]) -> torch.Tensor:
        """
        Extract descriptors from a batch of images.

        Args:
            images: List of (H, W, 3) numpy arrays (BGR uint8).

        Returns:
            Tensor of shape (N, 8448), L2-normalized.
        """
        tensors = [self._preprocess(img) for img in images]
        batch = torch.stack(tensors).to(self.device)
        descriptors = self.model(batch)  # (N, 8448), already L2-normalized
        return descriptors

    @torch.no_grad()
    def extract_single(self, image: np.ndarray) -> torch.Tensor:
        """Extract descriptor from a single image. Returns (8448,) tensor."""
        tensor = self._preprocess(image).unsqueeze(0).to(self.device)
        descriptor = self.model(tensor)  # (1, 8448)
        return descriptor.squeeze(0)
