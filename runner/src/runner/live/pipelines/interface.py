from asyncio import Task
from abc import ABC, abstractmethod

import torch
from pydantic import BaseModel, Field

from ..trickle import VideoFrame, VideoOutput, DEFAULT_WIDTH, DEFAULT_HEIGHT


class BaseParams(BaseModel):
    """
    Base parameters common to all pipelines.

    Includes shared image dimensions and UI behavior used by the orchestrating
    process (e.g., loading overlay while (re)initializing pipelines).
    """

    width: int = Field(
        default=DEFAULT_WIDTH,
        ge=384,
        le=1024,
        multiple_of=64,
        description="Output image width in pixels. Must be divisible by 64 and between 384-1024.",
    )

    height: int = Field(
        default=DEFAULT_HEIGHT,
        ge=384,
        le=1024,
        multiple_of=64,
        description="Output image height in pixels. Must be divisible by 64 and between 384-1024.",
    )

    show_reloading_frame: bool = Field(
        default=True,
        description="Whether to render a loading overlay while the pipeline initializes or reloads.",
    )

    def get_output_resolution(self) -> tuple[int, int]:
        """
        Get the output resolution as a (width, height) tuple. Sub-classes may override this method if the pipeline
        supports changing resolution during inference.
        """
        return (self.width, self.height)


class Pipeline(ABC):
    """Base class for all frame processing pipelines.

    Only ``transform`` is required. All other hooks have sane defaults
    and can be overridden as needed.

    Lifecycle:
        1. ``prepare_models()`` — called at build time (model download, compile)
        2. ``on_ready(**params)`` — called once at startup with initial params
        3. ``transform(frame, params)`` — called per frame
        4. ``on_update(**params)`` — called when params change mid-stream
        5. ``on_stop()`` — called on shutdown

    The framework manages frame queues, locking, and async/sync dispatch
    automatically. Pipelines just process frames.

    Example::

        @pipeline(name="my-pipeline", params=MyParams)
        class MyPipeline(Pipeline):
            def on_ready(self, **params):
                self.model = load_model()

            def transform(self, frame: VideoFrame, params: MyParams) -> torch.Tensor:
                return self.model(frame.tensor)
    """

    @abstractmethod
    def transform(
        self, frame: VideoFrame, params: BaseParams
    ) -> torch.Tensor | VideoOutput:
        """Process a single video frame. Called for every incoming frame.

        Args:
            frame: Input video frame with tensor (B, H, W, C) in [-1.0, 1.0].
            params: Current pipeline parameters.

        Returns:
            A ``torch.Tensor`` with the same layout as ``frame.tensor``,
            or a ``VideoOutput`` instance.
        """
        ...

    def on_ready(self, **params) -> None:
        """Called once after the pipeline is initialized with its first params.

        Use this to allocate models, move tensors to the correct device, etc.

        Args:
            **params: The initial pipeline parameters as keyword arguments.
        """

    def on_update(self, **params) -> Task[None] | None:
        """Called when pipeline parameters change mid-stream.

        If the update will take a long time (e.g. reloading the pipeline),
        return a Task that will be awaited by the caller (a loading overlay
        will be shown in the meantime).

        Args:
            **params: The updated pipeline parameters as keyword arguments.

        Returns:
            None for immediate updates, or an asyncio Task for long reloads.
        """
        return None

    def on_stop(self) -> None:
        """Called once when the pipeline is shutting down.

        Use this to release resources, close files, etc.
        """

    @classmethod
    def prepare_models(cls) -> None:
        """Download and/or compile any assets required by this pipeline.

        Called at build time (e.g., during Docker image build), not at runtime.
        """
