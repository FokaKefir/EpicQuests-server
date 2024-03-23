from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from utils import *
import os
import uuid

# input and generated path
input_path = os.getcwd() + '/data/input/'
gen_path = os.getcwd() + '/data/generated/'
model_path = os.getcwd() + '/models/'

# import model
model = get_model(2, model_path + 'im3vx6dd_generator.keras')

# create app
app = FastAPI()

# add cors
origins = [
    "http://localhost",
    "http://localhost:2000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# image generator call
def generate_image(image_name: str):
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

# file uploader
@app.post('/upload/')
async def upload_file(file: UploadFile = File(...)):

    file.filename = f'{uuid.uuid4()}.jpg'
    contents = await file.read()

    # save the file
    with open(f'{input_path}{file.filename}', 'wb') as fout:
        fout.write(contents)

    return {'filename': file.filename}  

# file uploader and generator
@app.post('/upload_and_generate/')
async def upload_file_and_generate_image(file: UploadFile = File(...)):

    file.filename = f'{uuid.uuid4()}.jpg'
    contents = await file.read()

    # save the file
    with open(f'{input_path}{file.filename}', 'wb') as fout:
        fout.write(contents)
    
    # generate image
    generate_image(file.filename)

    return {'filename': file.filename}  


# return generated image
@app.get('/generated_image/{image_name}')
async def get_generated_image(image_name: str):
    path = gen_path + image_name
    return FileResponse(path)