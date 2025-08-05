from fastapi import FastAPI, HTTPException
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
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
    response = client.responses.create(
        model="gpt-4o-mini",
        messages=[
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


class Horizontal(BaseModel):
    value: str
    desc: str


class Vertical(BaseModel):
    value: str
    desc: str


class Dimension(BaseModel):
    horizontal_dimensions: Horizontal
    vertical_dimensions: Vertical


class Drawing(BaseModel):
    drawing_number: str
    object_description: str
    dimensions: Dimension


class JsonExample2(BaseModel):
    drawings: list[Drawing]


class ImageRequest2(BaseModel):
    images: list[str]


@app.post("/callout")
async def process_callouts(req: ImageRequest2):
    base64_image = req.images[0].strip()
    base64_image2 = req.images[1].strip()
    logger.info("creating request for model...")
    response = client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": """Objective: Extract ALL characteristics and dimensions of specified objects from architectural drawings with absolute accuracy, suitable for product catalog matching. No errors in values or their assignments are permissible.
    Input Data:
        Image 1 (Plan View): This file contains a general floor plan of a room with various elements, including detailed section/elevation callouts.
        Image 2 (Elevation/Section Sheets): This file contains several detailed views (elevations or sections) of different objects, each identified by a unique number and name.

    Task:
        1.  **Identify All Target Objects and Corresponding View Numbers:**
            * On Image 1 (Plan View): Locate all graphic callouts. Each callout will appear as a circle divided into four equal parts, with numbers inside these parts, and black-filled arrows extending from the circle's arc, completing a right angle.
            * For each identified callout, **EXTREMELY ACCURATELY** extract ALL numerical values visible inside the circle (e.g., '10' and '9'). The top-left number within the circle should be used as the primary "drawing_number" for the output JSON.
            * Additionally, identify the drawing sheet number explicitly stated next to each callout (e.g., 'A-705').
            * Confirm the object or area that each callout's arrow(s) unequivocally point to on the Plan View (e.g., "kitchen countertop with double sink", "wall panel", "reception counter", "workstation").

        2.  **Match with Detailed Views and Extract Dimensions – Characteristics:**
            * For each identified callout and its corresponding primary view number (from step 1), find the detailed view on Image 2 (Elevation/Section Sheets) that perfectly matches this number (e.g., if the primary number is '7', look for a view labeled '7 [Room/Object Name]').
            * Confirm that the object in this detailed view is indeed the same object/area as indicated on the plan.
            * For each identified object, extract **ALL** visible numerical dimensions from **BOTH** the Plan View (Image 1) and the corresponding Elevation/Section View(s) (Image 2).
            * For each extracted dimension:
        * **STATE ITS VALUE WITH EXTREME PRECISION.**
        * Provide a **BRIEF, CONCISE EXPLANATION** of what the dimension represents, focusing on *what* it is a dimension *of* and *its type* (e.g., "countertop length", "sink depth", "height from floor to countertop", "upper cabinet height"). Avoid lengthy descriptions or specific start/end points of the dimension line.
        * For vertical dimensions, **SPECIAL ATTENTION**: Be aware that they may be written vertically from bottom to top. Carefully analyze the orientation of the numbers and lines.
        * **Rule for Vertical Dimension Chains**: If the drawing contains a chain of segmented vertical dimensions (e.g., 2'-3", 3") and a total sum height might also be indicated, **PRIORITIZE** the explicit numerical values of individual segments. Extract all segmented dimensions. DO NOT attempt to sum them yourself unless the total sum is also provided as a separate, clear, and unambiguous dimension line.
        * If there are other objects (e.g., appliances, other furniture elements) with clearly dimensioned sizes next to the main object on the plan or detailed view, extract their dimensions as well. Include them in the JSON, with a brief explanation, similar to the main object's dimensions.
        * If any dimension is not clearly legible or its assignment is uncertain, explicitly state that the dimension cannot be extracted with the required accuracy in its brief explanation field.

    Output Format:
    Answer must contain only JSON without additional explanations.
    The JSON should be an array of objects. Each object in the array represents a single identified object/view from the drawings. Each object should have the following structure:
        * A top-level key drawing_number indicating the primary view number (the top-left number inside the circular callout from the plan).
        * A key object_description providing a concise description of the object/area.
        * A key dimensions containing logically and hierarchically structured dimensions. This can include sub-keys like horizontal_dimensions, vertical_dimensions, and other_object_dimensions (for clearly dimensioned adjacent items).
    Do NOT include sheet numbers (like A-705), view types, or other meta-information beyond the initial drawing_number and object_description within each object's main structure. All keys and values within the JSON (except for the dimension values themselves, which are in imperial/metric units) should be in English.

    MANDATORY: Before providing the final JSON, THOROUGHLY DOUBLE-CHECK every numerical value and its brief explanation against the drawing.
        The response must contain JSON only.
        No explanations, introductions, descriptions, comments, apologies, or summaries should appear before or after the JSON.
        The response must start with [ and end with ], with absolutely nothing outside the JSON structure."""},
                    {
                        "type": "input_image",
                        "image_url": base64_image,
                        "detail": "high"
                    },
                    {"type": "input_image",
                     "image_url": base64_image2,
                     "detail": "high"}
                ]
            }
        ],
        text_format=JsonExample2
    )
    output = response.output_parsed
    logger.info("preparing response...")
    return output
