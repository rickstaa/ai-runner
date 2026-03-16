"""Depth Estimation (MiDaS) -- @pipeline example with model loading.

Uses MiDaS Small to produce a real-time depth map from video frames.
Demonstrates prepare_models (download + load) and GPU inference in transform.

Usage:
    python -m runner.live.infer \
        --pipeline '{
            "pipeline_cls": "examples.pipelines.depth_midas:DepthMidasPipeline"
        }'
"""

import logging

import torch
from pydantic import Field

from runner.live.pipelines import pipeline, BaseParams
from runner.live.trickle import VideoFrame


class DepthParams(BaseParams):
    """Parameters for depth estimation, adjustable mid-stream."""

    colormap: bool = Field(
        default=True,
        description="Apply colormap to depth output. False = grayscale.",
    )
    near_clip: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Clip depth values below this threshold (0.0 = no clip).",
    )
    far_clip: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Clip depth values above this threshold (1.0 = no clip).",
    )


@pipeline(name="depth-midas", params=DepthParams)
class DepthMidas:
    """Real-time depth estimation using MiDaS Small.

    Demonstrates:
    - prepare_models: downloads the model at startup
    - on_ready: loads model to GPU
    - transform: runs inference per frame
    - on_update: recomputes clip range when params change mid-stream
    """

    def on_ready(self, **params):
        self._near = params.get("near_clip", 0.0)
        self._far = params.get("far_clip", 1.0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True)
        self.model.to(self.device).eval()

        self.transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
        logging.info(f"MiDaS Small loaded on {self.device}")

    def transform(self, frame: VideoFrame, params: DepthParams) -> torch.Tensor:
        """Run MiDaS depth estimation on each frame."""
        # frame.tensor is (B, H, W, C) in [-1.0, 1.0]
        tensor = frame.tensor[0]  # (H, W, C), single batch
        h, w = tensor.shape[:2]

        # Convert to uint8 RGB for MiDaS transforms
        img = ((tensor + 1.0) / 2.0 * 255).byte().cpu().numpy()

        # MiDaS expects (B, C, H, W) float32
        input_batch = self.transforms(img).to(self.device)

        with torch.no_grad():
            depth = self.model(input_batch)  # (1, h', w')

        # Resize back to original resolution
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
        ).squeeze()  # (H, W)

        # Normalize to [0, 1] and apply depth clipping
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth = torch.clamp(depth, self._near, self._far)
        depth = (depth - self._near) / (self._far - self._near + 1e-8)

        if params.colormap:
            # Simple colormap: near=warm, far=cool
            r = depth
            g = 1.0 - torch.abs(depth - 0.5) * 2.0
            b = 1.0 - depth
            out = torch.stack([r, g, b], dim=-1)  # (H, W, 3)
        else:
            # Grayscale
            out = depth.unsqueeze(-1).expand(-1, -1, 3)  # (H, W, 3)

        # Convert to [-1, 1] and add batch dim → (1, H, W, C)
        out = (out * 2.0 - 1.0).unsqueeze(0)
        return out.to(frame.tensor.device)

    def on_update(self, **params):
        """Update clip range when params change mid-stream via control_url."""
        self._near = params.get("near_clip", 0.0)
        self._far = params.get("far_clip", 1.0)
        logging.info(f"Depth clip range updated: [{self._near}, {self._far}]")

    def on_stop(self):
        logging.info("DepthMidas stopped")

    @classmethod
    def prepare_models(cls):
        """Download MiDaS Small weights ahead of time."""
        torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True)
        logging.info("MiDaS Small weights downloaded")


DepthMidasPipeline = DepthMidas
