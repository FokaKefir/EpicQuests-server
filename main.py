from fastapi import FastAPI
from utils import *
import os

# input and generated path
input_path = os.getcwd() + '/data/input/'
gen_path = os.getcwd() + '/data/generated/'
model_path = os.getcwd() + '/models/'

# import model
model = get_model(2, model_path + 'im3vx6dd_generator.keras')

# create app
app = FastAPI()

# image generator call
@app.get("/generate/{image_name}")
async def generate_image(image_name: str):
    # load image
    image = load_image(input_path + image_name)

    # resize
    image = resize(image, IMG_HEIGHT, IMG_WIDTH)

    # normalize to [-1, 1]
    image = normalize(image)

    # generate the image
    pred = model(tf.expand_dims(image, 0))
    gen_image = pred[0]

    # save the generated image
    save_image(gen_image, gen_path + image_name)

    # return url to generated image
    return {"image_path": f"fokakefir.go.ro/generated/{image_name}"}