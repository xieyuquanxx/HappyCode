from io import BytesIO

from PIL import Image


def image_bytes2PIL(image_bytes) -> Image.Image:
    image = Image.open(BytesIO(image_bytes))
    return image
