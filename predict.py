import json
import os
import base64
from io import BytesIO
from typing import List

import torch
import torch.nn as nn
import numpy as np
from cog import BasePredictor, Input, Path
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint_legacy import StableDiffusionInpaintPipelineLegacy
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from PIL import Image
from transformers import CLIPFeatureExtractor

# Add this import for PIL ImageOps
import PIL.ImageOps

MODEL_ID = "stabilityai/stable-diffusion-2-1"
MODEL_CACHE = "diffusers-cache"
SAFETY_MODEL_CACHE = "diffusers-cache"
SAFETY_MODEL_ID = "CompVis/stable-diffusion-safety-checker"

if not os.path.exists("weights"):
    raise ValueError("dreambooth weights not found")

DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512
DEFAULT_SCHEDULER = "DDIM"

# grab instance_prompt from weights,
# unless empty string or not existent

DEFAULT_PROMPT = None
try:
    with open("weights/args.json") as f:
        args = json.load(f)
        DEFAULT_PROMPT = args["instance_prompt"]
except:
    pass
if not DEFAULT_PROMPT:
    DEFAULT_PROMPT = "a photo of an astronaut riding a horse on mars"

SAFETY_MODEL_CACHE = "diffusers-cache"
SAFETY_MODEL_ID = "CompVis/stable-diffusion-safety-checker"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading Safety pipeline...")
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_MODEL_ID,
            cache_dir=SAFETY_MODEL_CACHE,
            torch_dtype=torch.float16,
            local_files_only=True,
        ).to("cuda")
        feature_extractor = CLIPFeatureExtractor.from_pretrained(
            "openai/clip-vit-base-patch32", cache_dir=SAFETY_MODEL_CACHE
        )

        print("Loading SD pipeline...")
        self.txt2img_pipe = StableDiffusionPipeline.from_pretrained(
            "weights",
            safety_checker=self.safety_checker,
            feature_extractor=feature_extractor,
            torch_dtype=torch.float16,
        ).to("cuda")

        self.txt2img_pipe.unet.config.in_channels = 9
        unet_config = self.txt2img_pipe.unet.config
        unet_config["in_channels"] = 9

        self.img2img_pipe = StableDiffusionImg2ImgPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            safety_checker=self.txt2img_pipe.safety_checker,
            feature_extractor=self.txt2img_pipe.feature_extractor,
        ).to("cuda")

        # Add this setup code for inpainting_pipe
        print("Loading Inpainting pipeline...")
        self.inpainting_pipe = StableDiffusionInpaintPipelineLegacy(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            safety_checker=self.txt2img_pipe.safety_checker,
            feature_extractor=self.txt2img_pipe.feature_extractor,
        ).to("cuda")

        self.inpainting_pipe.unet.config.in_channels = 9

    # Add additional inpainting-related inputs to the predict function
    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default=DEFAULT_PROMPT,
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default=None,
        ),
        image: Path = Input(
            description="A starting image from which to generate variations (aka 'img2img'). If this input is set, the `width` and `height` inputs are ignored and the output will have the same dimensions as the input image.",
            default=None,
        ),
        width: int = Input(
            description="Width of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576,
                     640, 704, 768, 832, 896, 960, 1024],
            default=DEFAULT_WIDTH,
        ),
        height: int = Input(
            description="Height of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576,
                     640, 704, 768, 832, 896, 960, 1024],
            default=DEFAULT_HEIGHT,
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using init image. 1.0 corresponds to full destruction of information in init image",
            default=0.8,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        scheduler: str = Input(
            default=DEFAULT_SCHEDULER,
            choices=[
                "DDIM",
                "K_EULER",
                "DPMSolverMultistep",
                "K_EULER_ANCESTRAL",
                "PNDM",
                "KLMS",
            ],
            description="Choose a scheduler",
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        disable_safety_check: bool = Input(
            description="Disable safety check. Use at your own risk!", default=False
        ),
        mode: str = Input(
            description="Choose the mode of operation: 'txt2img', 'img2img', or 'inpaint'.",
            choices=["txt2img", "img2img", "inpaint"],
            default="inpaint",
        ),
        mask: Path = Input(
            description="Black and white image to use as mask. Required only in 'inpaint' mode. White pixels are inpainted and black pixels are preserved.",
            default=None,
        ),
        invert_mask: bool = Input(
            description="If this is true, then black pixels are inpainted and white pixels are preserved. Used only in 'inpaint' mode.",
            default=False,
        ),
        # ...
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if width * height > 786432:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
            )

        if mode == "inpaint":
            print("using inpaint")
            pipe = self.inpainting_pipe

            # Load the mask image and invert it if needed
            image = Image.open(image).convert("RGB")
            mask = Image.open(mask).convert("RGB")

            if invert_mask:
                mask = PIL.ImageOps.invert(mask)

            if image.width % 8 != 0 or image.height % 8 != 0:
                if mask.size == image.size:
                    mask = crop(mask)
                image = crop(image)

            if mask.size != image.size:
                print(
                    f"WARNING: Mask size ({mask.width}, {mask.height}) is different to image size ({image.width}, {image.height}). Mask will be resized to image size."
                )
                mask = mask.resize(image.size)

            extra_kwargs = {
                "init_image": image,
                "mask_image": mask,
            }
        elif image is not None:
            print("using img2img")
            pipe = self.img2img_pipe
            extra_kwargs = {
                "image": Image.open(image).convert("RGB"),
                "strength": prompt_strength,
            }
        else:
            print("using txt2img")
            pipe = self.txt2img_pipe
            extra_kwargs = {
                "width": width,
                "height": height,
            }

        pipe.scheduler = make_scheduler(scheduler, pipe.scheduler.config)

        generator = torch.Generator("cuda").manual_seed(seed)

        if disable_safety_check:
            pipe.safety_checker = None
        else:
            pipe.safety_checker = self.safety_checker

            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_outputs,
                guidance_scale=guidance_scale,
                generator=generator,
                num_inference_steps=num_inference_steps,
                **extra_kwargs,
            )

            samples = [
                output.images[i]
                for i, nsfw_flag in enumerate(output.nsfw_content_detected)
                if not nsfw_flag
            ]

            if len(samples) == 0:
                raise Exception(
                    f"NSFW content detected. Try running it again, or try a different prompt."
                )

            if num_outputs > len(samples):
                print(
                    f"NSFW content detected in {num_outputs - len(samples)} outputs, returning the remaining {len(samples)} images."
                )
            output_paths = []
            for i, sample in enumerate(samples):
                output_path = f"/tmp/out-{i}.png"
                sample.save(output_path)
                output_paths.append(Path(output_path))

            if len(output_paths) == 0:
                raise Exception(
                    f"NSFW content detected. Try running it again, or try a different prompt."
                )

            return output_paths


def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]


def crop(image):
    height = (image.height // 8) * 8
    width = (image.width // 8) * 8
    left = int((image.width - width) / 2)
    right = left + width
    top = int((image.height - height) / 2)
    bottom = top + height
    image = image.crop((left, top, right, bottom))
    return image
