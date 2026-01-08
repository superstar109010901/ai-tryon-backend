"""
Main FastAPI application for Virtual Clothing Try-On
This backend handles image uploads, processes them with SDXL models,
and returns generated images with different clothing styles.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from contextlib import asynccontextmanager
from typing import Optional
import uvicorn
import os
import sys
from pathlib import Path

# Add the backend directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from image_processor import ImageProcessor
from model_manager import ModelManager

# Global flag to track if models are loaded
models_loaded = False
model_manager = None
image_processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager: Initialize models on startup, cleanup on shutdown.
    This ensures models are ready before processing requests.
    """
    global models_loaded, model_manager, image_processor
    
    # Startup: Initialize models
    try:
        print("=" * 60)
        print("Initializing Virtual Try-On Backend for GPU Server...")
        print("=" * 60)
        
        model_manager = ModelManager()
        image_processor = ImageProcessor(model_manager)
        
        # Check and download models if needed
        print("Checking model availability...")
        await model_manager.ensure_models_available()
        
        # Preload models to GPU immediately (optimized for GPU servers)
        # This ensures models are ready when first request arrives
        print("Loading models to GPU...")
        model_manager.load_models()
        
        models_loaded = True
        print("=" * 60)
        print("✅ Models loaded successfully on GPU!")
        print("✅ Server ready to process requests")
        print("=" * 60)
    except Exception as e:
        print(f"⚠️  Warning: Could not load models: {e}")
        print("Server will run in placeholder mode for testing")
        models_loaded = False
        # Still initialize managers for placeholder mode
        if model_manager is None:
            model_manager = ModelManager()
            image_processor = ImageProcessor(model_manager)
    
    yield  # Application runs here
    
    # Shutdown: Cleanup (optional)
    if model_manager:
        model_manager.unload_models()
        print("Models unloaded")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Virtual Clothing Try-On API",
    description="API for generating virtual try-on images using SDXL",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS to allow frontend to connect
# In production, replace "*" with your frontend URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins - change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "message": "Virtual Clothing Try-On API",
        "status": "running",
        "models_loaded": models_loaded
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify server and model status.
    Frontend can call this to check if models are ready.
    """
    gpu_available = False
    if model_manager:
        gpu_available = model_manager.check_gpu_available()
    
    return {
        "status": "healthy",
        "models_loaded": models_loaded,
        "gpu_available": gpu_available
    }


@app.post("/generate")
async def generate_tryon(
    file: UploadFile = File(..., description="Person image to process"),
    style: str = Form(..., description="Clothing style: casual, formal, sportswear, streetwear, or random")
):
    """
    Main endpoint for generating virtual try-on images.
    
    Process:
    1. Receives uploaded person image and selected style
    2. Processes image with SDXL models
    3. Returns generated image with new clothing
    
    Args:
        file: Uploaded image file (JPEG, PNG)
        style: Selected clothing style
    
    Returns:
        JSON with generated image data or error message
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Validate style
        valid_styles = ["casual", "formal", "sportswear", "streetwear", "random"]
        style_lower = style.lower()
        if style_lower not in valid_styles:
            raise HTTPException(
                status_code=400, 
                detail=f"Style must be one of: {', '.join(valid_styles)}"
            )
        
        # Read uploaded image
        image_data = await file.read()
        
        # Process image with SDXL
        # If models aren't loaded, use placeholder mode
        if not models_loaded:
            # Return placeholder response for testing without GPU
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Placeholder mode - models not loaded",
                    "style": style_lower,
                    "image_url": None,
                    "placeholder": True
                }
            )
        
        # Process the image with actual SDXL models
        if image_processor is None:
            raise HTTPException(status_code=503, detail="Image processor not initialized")
        
        result = await image_processor.process_image(image_data, style_lower)
        
        if result["success"]:
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "style": style_lower,
                    "image_url": result["image_url"],
                    "message": "Image generated successfully"
                }
            )
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Processing failed"))
            
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
    """
    try:
        file_path = Path("temp_images") / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        
        return FileResponse(
            path=str(file_path),
            media_type="image/png",
            filename=filename
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")


if __name__ == "__main__":
    # Run the server
    # Optimized for GPU server deployment (Vast.ai, RunPod, etc.)
    # host="0.0.0.0" allows external connections from frontend
    # reload=False for production (better performance on GPU servers)
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # Accept connections from any IP (required for remote access)
        port=8080,        # Using port 8080 (maps to external port 36602 on Vast.ai)
        reload=False      # Disable reload for production GPU servers (better performance)
    )

