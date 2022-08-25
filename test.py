# make sure you're logged in with `huggingface-cli login`
from torch import autocast
from diffusers import StableDiffusionPipeline
from PIL import Image

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


pipe = StableDiffusionPipeline.from_pretrained(
	"CompVis/stable-diffusion-v1-4", 
    height=1024,
    weight=1024,
	use_auth_token=True
).to("cuda")

with autocast("cuda"):
    num_images = 4
    prompt = ["a photograph of an astronaut riding a horse"] * num_images

    images = pipe(prompt)["sample"]

    grid = image_grid(images, rows=2, cols=2)

    
grid.save("astronaut_rides_horse.png")