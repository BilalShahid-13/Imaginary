import torch
from diffusers import DiffusionPipeline

if torch.cuda.is_available():
    print("CUDA is available. GPU will be used.")
    device = "cuda"
else:
    print("CUDA is not available. CPU will be used.")
    device = "cpu"

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, device=device)

prompt = "batman"

images = pipe(prompt=prompt).images[0]

# Save the image to a file
images.save("image.png")

# Generate markdown image tag
with open("image.png", "rb") as f:
    img_data = f.read()

markdown_image_tag = f"![{prompt}]({'data:image/png;base64,' + img_data.decode('ascii')})"

print(markdown_image_tag)
