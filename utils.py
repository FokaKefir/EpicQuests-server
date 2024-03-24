import tensorflow as tf
from models import Generator, Generator2, Generator3
import cv2

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

    # filter noise
    median = cv2.medianBlur(image.numpy(), 3)

    # save image
    cv2.imwrite(path, cv2.cvtColor(median, cv2.COLOR_RGB2BGR))


def get_model(model_type, path):
    if model_type == 1:
        model = Generator()
    elif model_type == 2:
        model = Generator2()
    elif model_type == 3:
        model = Generator3()
        IMG_HEIGHT, IMG_WIDTH = 512, 512
    model.load_weights(path)
    return model
