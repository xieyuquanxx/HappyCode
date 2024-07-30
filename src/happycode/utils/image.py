from io import BytesIO

from PIL import Image


def image_bytes2PIL(image_bytes) -> Image.Image:
    image = Image.open(BytesIO(image_bytes))
    return image


def load_pil_images_from_path(image_list: list[str]) -> list[Image.Image]:
    """
    Support file path.

    Args:
        image_list (List[str]): the list of image paths.
    Returns:
        pil_images (List[PIL.Image.Image]): the list of PIL images.
    """

    return [Image.open(image_data).convert("RGB") for image_data in image_list]
