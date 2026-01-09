"""
Image Processor for Virtual Try-On
Handles image processing and calls Vast.ai Stable Diffusion API for clothing generation.
"""

import io
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict
import logging
import uuid
from datetime import datetime

from vast_ai_client import VastAIClient
from config import CONTROLNET_ENABLED, CONTROLNET_MODULE, CONTROLNET_MODEL, CONTROLNET_WEIGHT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Static prompt for clothing replacement - fixed to white shirt
STATIC_PROMPT = """
white shirt,
realistic clothing replacement,
preserve body shape,
natural fabric folds,
photorealistic,
studio lighting,
high detail
"""




class ImageProcessor:
    """
    Processes images for virtual try-on using Vast.ai Stable Diffusion API.
    
    Workflow:
    1. Load and preprocess input image
    2. Send image to Vast.ai img2img API with static prompts
    3. Receive generated image
    4. Save and return result
    """
    
    def __init__(self, vast_ai_url: str):
        """
        Initialize ImageProcessor with Vast.ai API client.
        
        Args:
            vast_ai_url: Base URL of Vast.ai Stable Diffusion API
                         (e.g., "http://localhost:8081" or "http://74.48.140.178:36769")
        """
        self.vast_ai_client = VastAIClient(vast_ai_url)
        self.temp_dir = Path("temp_images")
        self.temp_dir.mkdir(exist_ok=True)
        
        logger.info(f"ImageProcessor initialized with Vast.ai URL: {vast_ai_url}")
    
    async def process_image(self, image_data: bytes) -> Dict:
        """
        Main processing function: takes image, sends to Vast.ai API, returns generated image.
        
        Args:
            image_data: Raw image bytes from upload
        
        Returns:
            Dictionary with success status and image URL or error message
        """
        try:
            # Load image from bytes
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            
            # Resize image if too large (SD works best with reasonable sizes)
            # Maintain aspect ratio
            max_size = 1024
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                logger.info(f"Resized image to {image.size}")
            
            # Generate image using Vast.ai img2img API
            # Using static prompt and fixed SD parameters
            generated_image = await self.vast_ai_client.generate_img2img(
                image=image,
                prompt=STATIC_PROMPT.strip(),
                denoising_strength=0.45,  # Fixed parameter
                steps=30,                  # Fixed parameter
                cfg_scale=7,               # Fixed parameter
                sampler_name="DPM++ 2M Karras",  # Fixed sampler
                controlnet_enabled=CONTROLNET_ENABLED,
                controlnet_module=CONTROLNET_MODULE if CONTROLNET_ENABLED else None,
                controlnet_model=CONTROLNET_MODEL if CONTROLNET_ENABLED else None,
                controlnet_weight=CONTROLNET_WEIGHT if CONTROLNET_ENABLED else 1.0
            )
            
            # Save generated image temporarily
            filename = f"generated_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}.png"
            filepath = self.temp_dir / filename
            generated_image.save(filepath, "PNG")
            
            # Clean up old files (keep only last 10)
            self.cleanup_old_files(keep_count=10)
            
            logger.info(f"Image processed successfully: {filename}")
            
            return {
                "success": True,
                "image_url": f"/download/{filename}",
                "filename": filename
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def cleanup_old_files(self, keep_count: int = 10):
        """
        Clean up old generated images to save disk space.
        Keeps only the most recent files.
        
        Args:
            keep_count: Number of recent files to keep
        """
        try:
            files = sorted(
                self.temp_dir.glob("generated_*.png"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            # Delete files beyond keep_count
            for file in files[keep_count:]:
                file.unlink()
                logger.info(f"Deleted old file: {file.name}")
        except Exception as e:
            logger.warning(f"Error cleaning up files: {e}")
