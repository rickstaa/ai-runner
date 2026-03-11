import logging
from typing import Annotated, Dict, Tuple, Union

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image

from runner.dependencies import get_pipeline
from runner.pipelines.base import Pipeline
from runner.routes.utils import (
    HTTPError,
    ImageToTextResponse,
    RESPONSES,
    check_auth_token,
    check_model_id,
    execute_pipeline,
    file_exceeds_max_size,
    http_error,
)

router = APIRouter()

logger = logging.getLogger(__name__)

# Pipeline specific error handling configuration.
PIPELINE_ERROR_CONFIG: Dict[str, Tuple[Union[str, None], int]] = {
    # Specific error types.
    "OutOfMemoryError": (
        "Out of memory error. Try reducing input image resolution.",
        status.HTTP_500_INTERNAL_SERVER_ERROR,
    )
}

# Extend shared RESPONSES with additional status codes for this route.
IMAGE_TO_TEXT_RESPONSES = {
    **RESPONSES,
    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE: {"model": HTTPError},
}


@router.post(
    "/image-to-text",
    response_model=ImageToTextResponse,
    responses=IMAGE_TO_TEXT_RESPONSES,
    description="Transform image files to text.",
    operation_id="genImageToText",
    summary="Image To Text",
    tags=["generate"],
    openapi_extra={"x-speakeasy-name-override": "imageToText"},
)
@router.post(
    "/image-to-text/",
    response_model=ImageToTextResponse,
    responses=IMAGE_TO_TEXT_RESPONSES,
    include_in_schema=False,
)
async def image_to_text(
    image: Annotated[
        UploadFile, File(description="Uploaded image to transform with the pipeline.")
    ],
    prompt: Annotated[
        str,
        Form(description="Text prompt(s) to guide transformation."),
    ] = "",
    model_id: Annotated[
        str,
        Form(description="Hugging Face model ID used for transformation."),
    ] = "",
    pipeline: Pipeline = Depends(get_pipeline),
    token: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
):
    if auth_error := check_auth_token(token):
        return auth_error

    if model_error := check_model_id(model_id, pipeline.model_id):
        return model_error

    if file_exceeds_max_size(image, 50 * 1024 * 1024):
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content=http_error("File size exceeds limit"),
        )

    image = Image.open(image.file).convert("RGB")
    result, error = execute_pipeline(
        pipeline,
        default_error_message="Image-to-text pipeline error.",
        custom_error_config=PIPELINE_ERROR_CONFIG,
        prompt=prompt,
        image=image,
    )
    if error:
        return error
    return ImageToTextResponse(text=result)
