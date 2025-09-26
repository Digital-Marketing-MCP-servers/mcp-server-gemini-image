import base64
import io
import uuid
import logging
import os

import PIL.Image

logger = logging.getLogger(__name__)

OUTPUT_IMAGE_PATH = os.getenv("OUTPUT_IMAGE_PATH") or os.path.expanduser("~/gen_image")

if not os.path.exists(OUTPUT_IMAGE_PATH):
    os.makedirs(OUTPUT_IMAGE_PATH)

def validate_base64_image(base64_string: str) -> bool:
    """Validate if a string is a valid base64-encoded image.

    Args:
        base64_string: The base64 string to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        # Try to decode base64
        image_data = base64.b64decode(base64_string)

        # Try to open as image
        with PIL.Image.open(io.BytesIO(image_data)) as img:
            logger.debug(
                f"Validated base64 image, format: {img.format}, size: {img.size}"
            )
            return True

    except Exception as e:
        logger.warning(f"Invalid base64 image: {str(e)}")
        return False

def save_image(base64_image: str):
    img_bytes = base64.b64decode(base64_image)
    filename = f"{uuid.uuid4()}.jpg"

    static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../static"))
    filepath = os.path.join(static_dir, filename)

    with open(filepath, "wb") as f:
        f.write(img_bytes)

    return {"image_url": f"http://127.0.0.1:8002/static/{filename}"}


