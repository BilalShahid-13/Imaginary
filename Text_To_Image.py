

!pip install diffusers --upgrade
!pip install invisible_watermark transformers accelerate safetensors

import torch
from diffusers import DiffusionPipeline
# from diffusers import StableDiffusionPipeline


if torch.cuda.is_available():
    print("CUDA is available. GPU will be used.")
else:
    print("CUDA is not available. Please switch to a GPU runtime.")
    # You might want to handle this situation differently based on your requirements



# You might want to adjust the parameters based on your requirements
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
# pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")


pipe.to("cuda")




prompt = "batman"  # Enhanced prompt
# prompt = "batman and catwoman"

images = pipe(prompt=prompt).images[0]
display(images)
