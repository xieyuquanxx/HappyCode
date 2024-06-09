from io import BytesIO

from PIL import Image


def image_bytes2PIL(image_bytes):
    image = Image.open(BytesIO(image_bytes))
    return image
