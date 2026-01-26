"""
Main FastAPI application for Virtual Clothing Try-On
This backend handles image uploads, sends them to Vast.ai Stable Diffusion API,
and returns generated images with clothing replacement.
"""

import re
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from contextlib import asynccontextmanager
import uvicorn
import os
import sys
from pathlib import Path
from typing import Optional

# Add the backend directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from image_processor import ImageProcessor
from config import VAST_AI_SD_URL, DATA_ROOT

# Global image processor instance
image_processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager: Initialize Vast.ai API client on startup.
    """
    global image_processor
    
    # Startup: Initialize Vast.ai API client
    try:
        print("=" * 60)
        print("Initializing Virtual Try-On Backend...")
        print("=" * 60)
        print(f"Vast.ai SD API URL: {VAST_AI_SD_URL}")
        
        image_processor = ImageProcessor(VAST_AI_SD_URL)
        
        print("=" * 60)
        print("✅ Backend initialized successfully!")
        print("✅ Ready to process requests via Vast.ai API")
        print("=" * 60)
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize: {e}")
        print("Server may not function correctly")
        image_processor = None
    
    yield  # Application runs here
    
    # Shutdown: Cleanup (optional)
    print("Shutting down backend...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Virtual Clothing Try-On API",
    description="API for generating virtual try-on images using Vast.ai Stable Diffusion API",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS to allow frontend to connect
# Frontend: Create React App (react-scripts) running on port 3000 by default
# Backend: FastAPI on port 8384 (maps to external 48277 on Vast.ai)
# Production: Frontend deployed on Vercel at https://ai-tryon-3ypg.vercel.app
# Note: allow_credentials=False since no authentication/cookies are used
origins = [
    "http://localhost:3000",                    # Create React App default dev server
    "http://127.0.0.1:3000",                    # Alternative localhost address
    "https://ai-tryon-3ypg.vercel.app",          # Production Vercel domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,                              # Specific origins for development and production
    allow_origin_regex=r"https://.*\.vercel\.app",      # Allow all Vercel deployments and preview branches (including ai-tryon-six.vercel.app)
    allow_credentials=False,                            # No auth/cookies needed - file upload only
    allow_methods=["GET", "POST", "OPTIONS"],           # Explicitly allow needed methods
    allow_headers=["*"],                                # Allow all headers (needed for multipart/form-data file uploads)
    expose_headers=["X-Process-Time"],                  # Expose custom headers to frontend if needed
)


def _sanitize_username(s: str) -> str:
    """Allow alphanumeric, underscore, hyphen. Default to 'anonymous' if empty/invalid."""
    if not s or not isinstance(s, str):
        return "anonymous"
    out = re.sub(r"[^\w\-]", "", s.strip())[:64]
    return out if out else "anonymous"


def _sanitize_filename(s: str, default: str = "image.png") -> str:
    """Use basename only, safe chars. Ensure an extension."""
    if not s or not isinstance(s, str):
        return default
    name = os.path.basename(s.strip())
    name = re.sub(r"[^\w.\-]", "_", name)
    if not name:
        return default
    if "." not in name:
        name = f"{name}.png"
    return name


def save_upload_to_undress(username: str, filename: str, data: bytes) -> Optional[Path]:
    """
    Save uploaded image copy to /data/images/$username/undress/$filename.ext.
    Creates directory if needed. Returns path if saved, None on error.
    """
    try:
        user = _sanitize_username(username)
        fname = _sanitize_filename(filename)
        base = Path(DATA_ROOT) / "images" / user / "undress"
        base.mkdir(parents=True, exist_ok=True)
        path = base / fname
        path.write_bytes(data)
        return path
    except Exception as e:
        print(f"Warning: could not save upload to undress: {e}")
        return None


@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "message": "Virtual Clothing Try-On API",
        "status": "running",
        "vast_ai_url": VAST_AI_SD_URL,
        "processor_ready": image_processor is not None
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify server and Vast.ai API connection.
    Frontend can call this to check if backend is ready.
    """
    return {
        "status": "healthy",
        "vast_ai_url": VAST_AI_SD_URL,
        "processor_ready": image_processor is not None
    }


@app.post("/generate")
async def generate_tryon(
    file: UploadFile = File(..., description="Person image to process"),
    username: str = Form("anonymous", description="User identifier for saving uploads"),
):
    """
    Main endpoint for generating virtual try-on images.
    
    Process Flow:
    1. Frontend sends image (base64/binary)
    2. Backend saves a copy to /data/images/$username/undress/$filename.ext
    3. Backend injects STATIC_PROMPT and fixed SD parameters
    4. Backend calls Vast.ai SD API (img2img)
    5. Stable Diffusion returns generated image
    6. Backend returns image to frontend
    
    Args:
        file: Uploaded image file (JPEG, PNG)
        username: Optional. Used for upload path; defaults to "anonymous".
    
    Returns:
        JSON with generated image data or error message
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read uploaded image
        image_data = await file.read()
        
        # Save copy to /data/images/$username/undress/$filename.ext
        save_upload_to_undress(username, file.filename or "image.png", image_data)
        
        # Validate image processor is initialized
        if image_processor is None:
            raise HTTPException(
                status_code=503, 
                detail="Image processor not initialized. Check Vast.ai API connection."
            )
        
        # Process the image via Vast.ai API
        # ImageProcessor handles: image → Vast.ai API → generated image
        result = await image_processor.process_image(image_data)
        
        if result["success"]:
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "image_url": result["image_url"],
                    "message": "Image generated successfully via Vast.ai API"
                }
            )
        else:
            raise HTTPException(
                status_code=500, 
                detail=result.get("error", "Processing failed")
            )
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in generate endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/download/{filename}")
async def download_image(filename: str):
    """
    Endpoint to download generated images.
    Frontend calls this to download the generated try-on images.
    
    Optimized for fast response with proper headers and streaming.
    """
    try:
        file_path = Path("temp_images") / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Determine content type based on file extension
        content_type = "image/png"
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
            content_type = "image/jpeg"
        elif filename.lower().endswith('.webp'):
            content_type = "image/webp"
        
        # Use FileResponse with streaming for efficient file serving
        return FileResponse(
            path=str(file_path),
            media_type=content_type,
            filename=filename,
            headers={
                "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
                "Accept-Ranges": "bytes",  # Support range requests
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")


if __name__ == "__main__":
    # Run the server
    # Backend acts as middleware between frontend and Vast.ai SD API
    # host="0.0.0.0" allows external connections from frontend
    # reload=False for production
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # Accept connections from any IP (required for remote access)
        port=8384,        # Backend internal port (maps to external port 48277 on Vast.ai)
        reload=False      # Disable reload for production
    )
