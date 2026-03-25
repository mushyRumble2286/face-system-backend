from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import cv2
import numpy as np
from detector import detect_face_shape

app = FastAPI()

# Crucial for React frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, change to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FramePayload(BaseModel):
    image: str

@app.post("/api/detect")
async def detect(payload: FramePayload):
    try:
        # Strip the data URI prefix if present
        encoded_data = payload.image.split(',')[1] if ',' in payload.image else payload.image
        
        # Decode base64 to OpenCV image
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"status": "Error decoding image"}

        # Run detection
        result = detect_face_shape(img)
        return result

    except Exception as e:
        return {"status": f"Server error: {str(e)}"}