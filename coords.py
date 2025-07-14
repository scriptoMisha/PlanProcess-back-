import json
from typing import List
from typing import Union
import logging

REQUIRED_KEYS = ("x1", "y1", "x2", "y2")
logger = logging.getLogger(__name__)

def is_valid_coords(coords: dict) -> bool:
    return isinstance(coords, dict) and all(k in coords for k in REQUIRED_KEYS)


def extract_coordinates(json_data: Union[str, dict,list]) -> List[List[int]]:
    # Если передали строку — распарсим её
    if isinstance(json_data, str):
        try:
            json_data = json.loads(json_data)
        except json.JSONDecodeError as e:
            logger.error(f"Error with json parsing: {e}")
            raise RuntimeError(f"Error with json parsing: {e}")

    # Теперь проверяем, что именно получилось после парсинга
    if isinstance(json_data, dict):
        sections = json_data.get("Sections", [])
    elif isinstance(json_data, list):
        sections = json_data
    else:
        logger.warning(f"Unknow json_data type: {type(json_data)}")
        raise RuntimeError(f"Unknow json_data type: {type(json_data)}")

    result = []
    for section in sections:
        if not isinstance(section, dict):
            continue

        coords = section.get("coordinates")

        if is_valid_coords(coords):
            result.append([coords[k] for k in REQUIRED_KEYS])
        else:
            logger.warning(f"Wrong coordinates: {section}")

    return result
