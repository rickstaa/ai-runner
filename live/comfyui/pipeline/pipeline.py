import asyncio
import logging
import os

import torch
from comfystream.client import ComfyStreamClient

from runner.live.pipelines import Pipeline
from runner.live.trickle import VideoFrame, VideoOutput
from .params import ComfyUIParams


COMFY_UI_WORKSPACE_ENV = "COMFY_UI_WORKSPACE"
WARMUP_RUNS = 1


class ComfyUI(Pipeline):
    def __init__(self):
        comfy_ui_workspace = os.getenv(COMFY_UI_WORKSPACE_ENV)
        self.client = ComfyStreamClient(cwd=comfy_ui_workspace)
        self.params: ComfyUIParams

    def on_ready(self, **params):
        """Initialize the ComfyUI pipeline with given parameters."""
        new_params = ComfyUIParams(**params)
        logging.info(f"Initializing ComfyUI Pipeline with prompt: {new_params.prompt}")
        # TODO: currently its a single prompt, but need to support multiple prompts
        # Note: ComfyStreamClient methods are sync, called via _invoke in the framework.
        self.client.set_prompts_sync([new_params.prompt])
        self.params = new_params

        # Warm up the pipeline
        dummy_frame = VideoFrame(None, 0, 0)
        dummy_frame.side_data.input = torch.randn(
            1, new_params.height, new_params.width, 3
        )

        for _ in range(WARMUP_RUNS):
            self.client.put_video_input(dummy_frame)
            _ = self.client.get_video_output_sync()
        logging.info("Pipeline initialization and warmup complete")

    def transform(self, frame: VideoFrame, params: ComfyUIParams):
        """Process a single frame through ComfyUI."""
        frame.side_data.input = frame.tensor
        self.client.put_video_input(frame)
        result_tensor = self.client.get_video_output_sync()
        return result_tensor

    def on_update(self, **params):
        update_task = asyncio.create_task(self._do_update_params(**params))

        try:
            # If update completes quickly, return None (no reload needed).
            asyncio.get_event_loop().run_until_complete(
                asyncio.wait_for(asyncio.shield(update_task), timeout=2.0)
            )
            return None
        except asyncio.TimeoutError:
            logging.info("Update taking a while, returning task for loading overlay")
            return update_task

    async def _do_update_params(self, **params):
        """Perform the actual parameter update with logging and param setting."""
        new_params = ComfyUIParams(**params)
        if new_params == self.params:
            logging.info("No parameters changed")
            return

        logging.info(f"Updating ComfyUI Pipeline Prompt: {new_params.prompt}")
        # TODO: currently its a single prompt, but need to support multiple prompts
        try:
            await self.client.update_prompts([new_params.prompt])
        except Exception as e:
            logging.error(f"Error updating ComfyUI Pipeline Prompt: {e}")
            raise e
        self.params = new_params

    def on_stop(self):
        try:
            logging.info("Stopping ComfyUI pipeline")
            logging.info("Waiting for ComfyUI client to cleanup")
            # Note: cleanup is async on the client; the framework's _invoke handles it.
            self.client.cleanup_sync()
            logging.info("ComfyUI client cleanup complete")
            # Force CUDA cache clear
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        except Exception as e:
            logging.error(f"Error stopping ComfyUI pipeline: {e}")
        finally:
            self.client = None
            logging.info("ComfyUI pipeline stopped")

    @classmethod
    def prepare_models(cls):
        raise NotImplementedError(
            "ComfyUI uses a separate model preparation flow. "
            "See dl_checkpoints.sh download_comfyui_live_models()."
        )
