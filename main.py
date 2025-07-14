from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from logging_config import setup_logging
from coords import extract_coordinates
from crop_sections import *
import logging

setup_logging()
app = FastAPI()

origins = ["https://scriptoMisha.github.io"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# # Подключение к OpenAI (предпочтительно через переменную окружения)
client = OpenAI()

logger = logging.getLogger(__name__)


class ImageRequest(BaseModel):
    image_base64: str  # без "data:image/png;base64," — только сама строка base64


class Coordinates(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class Section(BaseModel):
    SectionName: str
    SectionScale: str
    coordinates: Coordinates


class JsonExample(BaseModel):
    Sections: list[Section]


@app.post("/process")
async def process_image(req: ImageRequest):
    base64_image = req.image_base64.strip()
    logger.info("creating request for model...")
    response = client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": """You are an assistant specialized in analyzing architectural or technical drawings. I will provide you with an image of a drawing. Your task is to detect and extract only Floor Plans.
        A Floor Plan is a scaled drawing showing a top-down view of a building (as if sliced horizontally). It must display:
        Layout of internal and external walls
        Doors and windows 
        Room names and dimensions
        
        Exclude:
            Section Views (vertical slices)
            Elevation Views (side views)
            Detail Callouts (zoomed-in fragments)
            MEP or system diagrams not integrated into the architectural plan

        Your task is to:
            Detect and extract only actual Floor Plans from the image.
            Extract and return the exact name or title shown in the drawing for each floor plan section — do not invent or guess names.
            Extract the scale if it's written near the floor plan (e.g., “1:100”).
            Return the bounding box coordinates of each detected floor plan section in the image.

        Return your response strictly as a JSON array, using this format:
        [
            {
                "section_name": "string",   // the exact text shown in the drawing
                "scale": "string",          // the actual scale written on the plan, if available
                "coordinates": {
                    "x1": int,
                    "y1": int,
                    "x2": int,
                    "y2": int
                }
            }
        ]
        Do not include any extra explanation or comments — only return valid JSON. If no valid floor plans are found, return an empty JSON array."""},
                    {
                        "type": "input_image",
                        "image_url": base64_image,
                        "detail": "high"
                    }
                ]
            }
        ],
        text_format=JsonExample,

    )
    output = response.output_parsed
    parsed_data = output.json()
    logger.info("extracting sections...")
    try:
        coords = extract_coordinates(parsed_data)
        sections_photo = crop_sections_from_base64(base64_image, coords)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail={e})

    logger.info("preparing response...")
    return JSONResponse(content={"metadata": parsed_data, "images": sections_photo})
