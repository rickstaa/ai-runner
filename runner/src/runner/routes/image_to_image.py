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
    "/image-to-image",
    response_model=ImageResponse,
    responses=RESPONSES,
    description="Apply image transformations to a provided image.",
    operation_id="genImageToImage",
    summary="Image To Image",
    tags=["generate"],
    openapi_extra={"x-speakeasy-name-override": "imageToImage"},
)
@router.post(
    "/image-to-image/",
    response_model=ImageResponse,
    responses=RESPONSES,
    include_in_schema=False,
)
async def image_to_image(
    prompt: Annotated[
        str,
        Form(description="Text prompt(s) to guide image generation."),
    ],
    image: Annotated[
        UploadFile,
        File(description="Uploaded image to modify with the pipeline."),
    ],
    model_id: Annotated[
        str,
        Form(description="Hugging Face model ID used for image generation."),
    ] = "",
    loras: Annotated[
        str,
        Form(
            description=(
                "A LoRA (Low-Rank Adaptation) model and its corresponding weight for "
                'image generation. Example: { "latent-consistency/lcm-lora-sdxl": '
                '1.0, "nerijs/pixel-art-xl": 1.2}.'
            )
        ),
    ] = "",
    strength: Annotated[
        float,
        Form(
            description=(
                "Degree of transformation applied to the reference image (0 to 1)."
            )
        ),
    ] = 0.8,
    guidance_scale: Annotated[
        float,
        Form(
            description=(
                "Encourages model to generate images closely linked to the text prompt "
                "(higher values may reduce image quality)."
            )
        ),
    ] = 7.5,
    image_guidance_scale: Annotated[
        float,
        Form(
            description=(
                "Degree to which the generated image is pushed towards the initial "
                "image."
            )
        ),
    ] = 1.5,
    negative_prompt: Annotated[
        str,
        Form(
            description=(
                "Text prompt(s) to guide what to exclude from image generation. "
                "Ignored if guidance_scale < 1."
            )
        ),
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
    ] = 100,  # NOTE: Hardcoded due to varying pipeline values.
    num_images_per_prompt: Annotated[
        int,
        Form(description="Number of images to generate per prompt."),
    ] = 1,
    pipeline: Pipeline = Depends(get_pipeline),
    token: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
):
    if auth_error := check_auth_token(token):
        return auth_error

    if model_error := check_model_id(model_id, pipeline.model_id):
        return model_error

    seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    seeds = [seed + i for i in range(num_images_per_prompt)]

    image = Image.open(image.file).convert("RGB")

    # TODO: Process one image at a time to avoid CUDA OEM errors. Can be removed again
    # once LIV-243 and LIV-379 are resolved.
    images = []
    has_nsfw_concept = []
    for seed in seeds:
        result, error = execute_pipeline(
            pipeline,
            default_error_message="Image-to-image pipeline error.",
            custom_error_config=PIPELINE_ERROR_CONFIG,
            prompt=prompt,
            image=image,
            strength=strength,
            loras=loras,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            negative_prompt=negative_prompt,
            safety_check=safety_check,
            seed=seed,
            num_images_per_prompt=1,
            num_inference_steps=num_inference_steps,
        )
        if error:
            return error
        imgs, nsfw_checks = result
        images.extend(imgs)
        has_nsfw_concept.extend(nsfw_checks)

    # TODO: Return None once Go codegen tool supports optional properties
    # OAPI 3.1 https://github.com/deepmap/oapi-codegen/issues/373
    output_images = [
        {"url": image_to_data_url(img), "seed": sd, "nsfw": nsfw or False}
        for img, sd, nsfw in zip(images, seeds, has_nsfw_concept)
    ]

    return {"images": output_images}
