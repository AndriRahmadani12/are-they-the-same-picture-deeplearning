from fastapi import FastAPI, UploadFile, File, HTTPException
from io import BytesIO
from app.image_comparator import ImageComparator
from app.schemas import ImageFiles
from PIL import Image
from typing import Dict

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/compare_images/")
async def compare_images(
    image1: UploadFile = File(...), 
    image2: UploadFile = File(...)
) -> Dict[str, float]:
    comparator = ImageComparator()
    # Validate file types
    allowed_types = {"image/jpeg", "image/png", "image/jpg"}
    if image1.content_type not in allowed_types or image2.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are supported"
        )

    try:
        image1_data = Image.open(BytesIO(await image1.read())).convert("RGB")
        image2_data = Image.open(BytesIO(await image2.read())).convert("RGB")
        similarity_score = comparator.calculate_similarity(image1_data, image2_data)
        
        return {"similarity_score": round(similarity_score, 4)}

    except IOError:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file format or corrupted image"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
