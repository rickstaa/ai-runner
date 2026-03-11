import logging
import time
from typing import Annotated, Dict, Tuple, Union

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from runner.dependencies import get_pipeline
from runner.pipelines.base import Pipeline
from runner.routes.utils import (
    AudioResponse,
    RESPONSES,
    check_auth_token,
    check_model_id,
    execute_pipeline,
    audio_to_data_url,
    http_error,
)

router = APIRouter()

logger = logging.getLogger(__name__)

# Pipeline specific error handling configuration.
PIPELINE_ERROR_CONFIG: Dict[str, Tuple[Union[str, None], int]] = {
    # Specific error types.
    "OutOfMemoryError": (
        "Out of memory error. Try reducing text input length.",
        status.HTTP_500_INTERNAL_SERVER_ERROR,
    )
}


class TextToSpeechParams(BaseModel):
    # TODO: Make model_id and other None properties optional once Go codegen tool
    # supports OAPI 3.1 https://github.com/deepmap/oapi-codegen/issues/373
    model_id: Annotated[
        str,
        Field(
            default="",
            description="Hugging Face model ID used for text to speech generation.",
        ),
    ]
    text: Annotated[
        str, Field(default="", description=("Text input for speech generation."))
    ]
    description: Annotated[
        str,
        Field(
            default=(
                "A male speaker delivers a slightly expressive and animated speech "
                "with a moderate speed and pitch."
            ),
            description=("Description of speaker to steer text to speech generation."),
        ),
    ]


@router.post(
    "/text-to-speech",
    response_model=AudioResponse,
    responses=RESPONSES,
    description=(
        "Generate a text-to-speech audio file based on the provided text input and "
        "speaker description."
    ),
    operation_id="genTextToSpeech",
    summary="Text To Speech",
    tags=["generate"],
    openapi_extra={"x-speakeasy-name-override": "textToSpeech"},
)
@router.post(
    "/text-to-speech/",
    response_model=AudioResponse,
    responses=RESPONSES,
    include_in_schema=False,
)
async def text_to_speech(
    params: TextToSpeechParams,
    pipeline: Pipeline = Depends(get_pipeline),
    token: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
):
    # Ensure required parameters are non-empty.
    # TODO: Remove if go-livepeer validation is fixed. Was disabled due to optional
    # params issue.
    if not params.text:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=http_error("Text input must be provided."),
        )

    if auth_error := check_auth_token(token):
        return auth_error

    if model_error := check_model_id(params.model_id, pipeline.model_id):
        return model_error

    start = time.time()
    result, error = execute_pipeline(
        pipeline,
        default_error_message="Text-to-speech pipeline error.",
        custom_error_config=PIPELINE_ERROR_CONFIG,
        params=params,
    )
    if error:
        return error
    logger.info(f"TextToSpeechPipeline took {time.time() - start} seconds.")

    return {"audio": {"url": audio_to_data_url(result)}}
