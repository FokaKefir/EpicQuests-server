import tensorflow as tf
from models import Generator, Generator2

IMG_HEIGHT = 256
IMG_WIDTH = 256

# resize image to the given height and width 
def resize(image, height, width):
    image = tf.image.resize(image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return image

# Normalize the image to [-1, 1]
def normalize(image):
    image = (image / 127.5) - 1
    return image

# load image from path
def load_image(image_file):
    # Read and decode an image file to a uint8 tensor
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image)

    # Convert image to float32 tensors
    image = tf.cast(image, tf.float32)

    return image

def save_image(image, path):
    # [-1, 1] -> [0, 256]
    image = (image + 1) * 127.5

    # convert image back to uint8 
    image = tf.cast(image, tf.uint8)

    # encode image to jpg
    image = tf.io.encode_jpeg(image)
    
    # write file
    tf.io.write_file(path, image)

def get_model(model_type, path):
    if model_type == 1:
        model = Generator()
    elif model_type == 2:
        model = Generator2()
    model.load_weights(path)
    return model
