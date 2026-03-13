from diffusers import DiffusionPipeline
from src.tools.prompt_utils import rewrite
import torch
import os
from dotenv import load_dotenv

load_dotenv()

has_api_key = "DASH_API_KEY" in os.environ or "DASHSCOPE_API_KEY" in os.environ
if not has_api_key:
    print("Warning: DASH_API_KEY not set. The image will be generated using the basic prompt without LM enhancement.")

# Initialize the pipeline
pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

# Generate with different aspect ratios
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472)
}

# English prompt (original)
original_prompt = "A cute little kitten sitting in a garden"

# Polish the prompt using our advanced rewriting system if API key is present
if has_api_key:
    polished_prompt = rewrite(original_prompt)
    print(f"Original: {original_prompt}\nPolished: {polished_prompt}")
else:
    polished_prompt = original_prompt
    print(f"Using basic prompt: {original_prompt}")

width, height = aspect_ratios["16:9"]

image = pipe(
    prompt=polished_prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cuda").manual_seed(42)
).images[0]

image.save("example.png")