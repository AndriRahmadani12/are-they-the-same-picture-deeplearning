from pydantic import BaseModel
from fastapi import UploadFile
class ImageFiles(BaseModel):
    image1: UploadFile
    image2: UploadFile