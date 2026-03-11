import logging
import traceback
from typing import Annotated, Any, Dict, Tuple, Union

import torch
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from runner.dependencies import get_pipeline
from runner.pipelines.base import Pipeline
from runner.routes.utils import (
    LiveVideoToVideoResponse,
    RESPONSES,
    check_auth_token,
    check_model_id,
    handle_pipeline_exception,
    http_error,
)

router = APIRouter()

logger = logging.getLogger(__name__)

# Pipeline specific error handling configuration.
PIPELINE_ERROR_CONFIG: Dict[str, Tuple[Union[str, None], int]] = {
    "OutOfMemoryError": (
        "Out of memory error. Try reducing input image resolution.",
        status.HTTP_500_INTERNAL_SERVER_ERROR,
    )
}

class LiveVideoToVideoParams(BaseModel):
    subscribe_url: Annotated[
        str,
        Field(
            ...,
            description="Source URL of the incoming stream to subscribe to.",
        ),
    ]
    publish_url: Annotated[
        str,
        Field(
            ...,
            description="Destination URL of the outgoing stream to publish.",
        ),
    ]
    control_url: Annotated[
        str,
        Field(
            default="",
            description="URL for subscribing via Trickle protocol for updates in the live video-to-video generation params.",
        ),
    ]
    events_url: Annotated[
        str,
        Field(
            default="",
            description="URL for publishing events via Trickle protocol for pipeline status and logs.",
        ),
    ]
    model_id: Annotated[
        str,
        Field(
            default="",
            description="Name of the pipeline to run in the live video to video job. Notice that this is named model_id for consistency with other routes, but it does not refer to a Hugging Face model ID. The exact model(s) depends on the pipeline implementation and might be configurable via the `params` argument."
        ),
    ]
    params: Annotated[
        Dict,
        Field(
            default={},
            description="Initial parameters for the pipeline."
        ),
    ]
    gateway_request_id: Annotated[
        str,
        Field(
            default="",
            description="The ID of the Gateway request (for logging purposes)."
        ),
    ]
    manifest_id: Annotated[
        str,
        Field(
            default="",
            description="The manifest ID from the orchestrator (for logging purposes)."
        ),
    ]
    stream_id: Annotated[
        str,
        Field(
            default="",
            description="The Stream ID (for logging purposes)."
        ),
    ]

@router.post(
    "/live-video-to-video",
    response_model=LiveVideoToVideoResponse,
    responses=RESPONSES,
    description="Apply transformations to a live video streamed to the returned endpoints.",
    operation_id="genLiveVideoToVideo",
    summary="Live Video To Video",
    tags=["generate"],
    openapi_extra={"x-speakeasy-name-override": "liveVideoToVideo"},
)
@router.post(
    "/live-video-to-video/",
    response_model=LiveVideoToVideoResponse,
    responses=RESPONSES,
    include_in_schema=False,
)
async def live_video_to_video(
    params: LiveVideoToVideoParams,
    pipeline: Pipeline = Depends(get_pipeline),
    token: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
):
    if auth_error := check_auth_token(token):
        return auth_error

    if model_error := check_model_id(params.model_id, pipeline.model_id):
        return model_error

    try:
        pipeline(**params.model_dump(), request_id=params.gateway_request_id)
    except Exception as e:
        if isinstance(e, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
        logger.error(f"LiveVideoToVideoPipeline error: {e}")
        logger.error(traceback.format_exc())
        return handle_pipeline_exception(
            e,
            default_error_message="live-video-to-video pipeline error.",
            custom_error_config=PIPELINE_ERROR_CONFIG,
        )

    # outputs unused for now; the orchestrator is setting these
    return {'publish_url':"", 'subscribe_url': "", 'control_url': ""}
