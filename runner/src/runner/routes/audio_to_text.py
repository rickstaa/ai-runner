import logging
from typing import Annotated, Dict, Tuple, Union

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from runner.dependencies import get_pipeline
from runner.pipelines.base import Pipeline
from runner.routes.utils import (
    HTTPError,
    TextResponse,
    RESPONSES,
    check_auth_token,
    check_model_id,
    execute_pipeline,
    file_exceeds_max_size,
    get_media_duration_ffmpeg,
    http_error,
    parse_key_from_metadata,
)

router = APIRouter()

logger = logging.getLogger(__name__)

# Pipeline specific error handling configuration.
AUDIO_FORMAT_ERROR_MESSAGE = "Unsupported audio format or malformed file."
PIPELINE_ERROR_CONFIG: Dict[str, Tuple[Union[str, None], int]] = {
    # Specific error types.
    "AudioConversionError": (
        AUDIO_FORMAT_ERROR_MESSAGE,
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    ),
    "Soundfile is either not in the correct format or is malformed": (
        AUDIO_FORMAT_ERROR_MESSAGE,
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    ),
}

# Extend shared RESPONSES with additional status codes for this route.
AUDIO_RESPONSES = {
    **RESPONSES,
    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE: {"model": HTTPError},
    status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {"model": HTTPError},
}


def parse_return_timestamps(value: str) -> Union[bool, str]:
    """Convert a string to a boolean or return the string as is. Sentence is considered
    True as it is the model default value.

    Args:
        value: The value to parse.

    Returns:
        The parsed value.
    """
    value_lower = value.lower()
    if value_lower in ("true", "1", "sentence"):
        return True
    if value_lower in ("false", "0"):
        return False
    return value_lower


@router.post(
    "/audio-to-text",
    response_model=TextResponse,
    responses=AUDIO_RESPONSES,
    description="Transcribe audio files to text.",
    operation_id="genAudioToText",
    summary="Audio To Text",
    tags=["generate"],
    openapi_extra={"x-speakeasy-name-override": "audioToText"},
)
@router.post(
    "/audio-to-text/",
    response_model=TextResponse,
    responses=AUDIO_RESPONSES,
    include_in_schema=False,
)
def audio_to_text(
    audio: Annotated[
        UploadFile, File(description="Uploaded audio file to be transcribed.")
    ],
    model_id: Annotated[
        str,
        Form(description="Hugging Face model ID used for transcription."),
    ] = "",
    pipeline: Pipeline = Depends(get_pipeline),
    return_timestamps: Annotated[
        str,
        Form(
            description=(
                "Return timestamps for the transcribed text. Supported values: "
                "'sentence', 'word', or a string boolean ('true' or 'false'). Default "
                "is 'true' ('sentence'). 'false' means no timestamps. 'word' means "
                "word-based timestamps."
            )
        ),
    ] = "true",
    metadata: Annotated[
        str,
        Form(description="Additional job information to be passed to the pipeline."),
    ] = "{}",
    token: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
):
    return_timestamps = parse_return_timestamps(return_timestamps)

    if auth_error := check_auth_token(token):
        return auth_error

    if model_error := check_model_id(model_id, pipeline.model_id):
        return model_error

    if file_exceeds_max_size(audio, 50 * 1024 * 1024):
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content=http_error("File size exceeds limit."),
        )

    try:
        duration = parse_key_from_metadata(metadata, "duration", float)
        if duration is None:
            logger.warning(
                f"duration not provided in request, calculating with ffprobe"
            )
            duration = get_media_duration_ffmpeg(audio.file.read())
            audio.file.seek(0)  # Reset file pointer
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=http_error("Unable to calculate duration of file"),
        )

    result, error = execute_pipeline(
        pipeline,
        default_error_message="Audio-to-text pipeline error.",
        custom_error_config=PIPELINE_ERROR_CONFIG,
        audio=audio,
        return_timestamps=return_timestamps,
        duration=duration,
    )
    if error:
        return error
    return result
