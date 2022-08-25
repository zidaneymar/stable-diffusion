import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import base64
from io import BytesIO
import os


# # make sure you're logged in with `huggingface-cli login`
# from torch import autocast
# from diffusers import StableDiffusionPipeline
from PIL import Image

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


# pipe = StableDiffusionPipeline.from_pretrained(
# 	"CompVis/stable-diffusion-v1-4", 
#     height=1024,
#     weight=1024,
# 	use_auth_token=True
# ).to("cuda")

# with autocast("cuda"):
#     num_images = 4
#     prompt = ["a photograph of an astronaut riding a horse"] * num_images

#     images = pipe(prompt)["sample"]

#     grid = image_grid(images, rows=2, cols=2)

    
# grid.save("astronaut_rides_horse.png")

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
    
    # this will substitute the default PNDM scheduler for K-LMS  
    lms = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")

    model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=lms, use_auth_token=HF_AUTH_TOKEN).to("cuda")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    with autocast("cuda"):
        images = model([prompt] * 4)["sample"]
        grid = image_grid(images, rows=2, cols=2)
    buffered = BytesIO()
    grid.save(buffered,format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Return the results as a dictionary
    return {'image_base64': image_base64}