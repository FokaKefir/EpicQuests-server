from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse

from random import randint
import uuid

import os

# input and generated path
input_path = os.getcwd() + '/data/input/'

# create app
app = FastAPI()

@app.post('/upload/')
async def upload_file(file: UploadFile = File(...)):

    file.filename = f'{uuid.uuid4()}.jpg'
    contents = await file.read()

    # save the file
    with open(f'{input_path}{file.filename}', 'wb') as fout:
        fout.write(contents)

    return {'filename': file.filename}

@app.get()