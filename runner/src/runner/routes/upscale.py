import logging
import random
from typing import Annotated, Dict, Tuple, Union

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image, ImageFile

from runner.dependencies import get_pipeline
from runner.pipelines.base import Pipeline
from runner.routes.utils import (
    ImageResponse,
    RESPONSES,
    check_auth_token,
    check_model_id,
    execute_pipeline,
    image_to_data_url,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

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


# TODO: Make model_id and other None properties optional once Go codegen tool supports
# OAPI 3.1 https://github.com/deepmap/oapi-codegen/issues/373
@router.post(
    "/upscale",
    response_model=ImageResponse,
    responses=RESPONSES,
    description="Upscale an image by increasing its resolution.",
    operation_id="genUpscale",
    summary="Upscale",
    tags=["generate"],
    openapi_extra={"x-speakeasy-name-override": "upscale"},
)
@router.post(
    "/upscale/",
    response_model=ImageResponse,
    responses=RESPONSES,
    include_in_schema=False,
)
async def upscale(
    prompt: Annotated[
        str,
        Form(description="Text prompt(s) to guide upscaled image generation."),
    ],
    image: Annotated[
        UploadFile,
        File(description="Uploaded image to modify with the pipeline."),
    ],
    model_id: Annotated[
        str,
        Form(description="Hugging Face model ID used for upscaled image generation."),
    ] = "",
    safety_check: Annotated[
        bool,
        Form(
            description=(
                "Perform a safety check to estimate if generated images could be "
                "offensive or harmful."
            )
        ),
    ] = True,
    seed: Annotated[int, Form(description="Seed for random number generation.")] = None,
    num_inference_steps: Annotated[
        int,
        Form(
            description=(
                "Number of denoising steps. More steps usually lead to higher quality "
                "images but slower inference. Modulated by strength."
            )
        ),
    ] = 75,  # NOTE: Hardcoded due to varying pipeline values.
    pipeline: Pipeline = Depends(get_pipeline),
    token: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
):
    if auth_error := check_auth_token(token):
        return auth_error

    if model_error := check_model_id(model_id, pipeline.model_id):
        return model_error

    seed = seed or random.randint(0, 2**32 - 1)

    image = Image.open(image.file).convert("RGB")

    result, error = execute_pipeline(
        pipeline,
        default_error_message="Upscale pipeline error.",
        custom_error_config=PIPELINE_ERROR_CONFIG,
        prompt=prompt,
        image=image,
        num_inference_steps=num_inference_steps,
        safety_check=safety_check,
        seed=seed,
    )
    if error:
        return error

    images, has_nsfw_concept = result
    seeds = [seed]

    # TODO: Return None once Go codegen tool supports optional properties
    # OAPI 3.1 https://github.com/deepmap/oapi-codegen/issues/373
    output_images = [
        {"url": image_to_data_url(img), "seed": sd, "nsfw": nsfw or False}
        for img, sd, nsfw in zip(images, seeds, has_nsfw_concept)
    ]

    return {"images": output_images}
