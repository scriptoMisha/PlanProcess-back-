import base64
from io import BytesIO
from PIL import Image
import logging

logger = logging.getLogger(__name__)
def crop_sections_from_base64(base64_image: str, coordinates: list[list[int]]) -> list[str]:
    if base64_image.startswith('data:image'):
        base64_image = base64_image.split(',', 1)[1]  # отделить и взять часть после запятой
    image_data = base64.b64decode(base64_image)
    image = Image.open(BytesIO(image_data))

    cropped_base64_list = []
    for section in coordinates:
        logger.info(f"type of section is: {type(section)}")
        if len(section) != 4:
            logger.error(f"Every section must have 4 coordinates: [x1, y1, x2, y2] in section: {section}")
            raise RuntimeError("Every section must have 4 coordinates: [x1, y1, x2, y2]")

        x1, y1, x2, y2 = section
        cropped = image.crop((x1, y1, x2, y2))

        buffer = BytesIO()
        cropped.save(buffer, format=image.format or 'PNG')
        cropped_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        cropped_base64_list.append(cropped_base64)

    return cropped_base64_list
