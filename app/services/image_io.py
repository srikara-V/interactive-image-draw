from io import BytesIO
from typing import Tuple
import base64

from PIL import Image


def image_to_data_url(image: Image.Image, image_format: str = "PNG") -> str:
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "image/png" if image_format.upper() == "PNG" else "image/jpeg"
    return "data:%s;base64,%s" % (mime, encoded)


def normalize_image(image: Image.Image, size: Tuple[int, int] = (768, 768)) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (18, 18, 20))
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def read_upload_image(raw: bytes) -> Image.Image:
    with BytesIO(raw) as buffer:
        image = Image.open(buffer)
        image.load()
    return normalize_image(image)
