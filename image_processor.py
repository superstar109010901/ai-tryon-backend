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

# Static prompt for clothing replacement
STATIC_PROMPT = "person wearing clean white shirt, realistic fabric, natural lighting"

# Negative prompt to avoid unwanted changes (especially face changes)
NEGATIVE_PROMPT = "face change, different face, deformed face, altered identity, different person, bad anatomy"




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
            
            # Resize image to match target dimensions (512x768)
            # This ensures consistent output size as specified
            target_width = 512
            target_height = 768
            if image.size != (target_width, target_height):
                image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
                logger.info(f"Resized image to {image.size}")
            
            # Generate mask for inpainting
            # White (255) = area SD CAN change (clothing)
            # Black (0) = area SD MUST NOT change (face, hair, body)
            mask = self.generate_clothing_mask(image)
            
            # Generate image using Vast.ai img2img API with inpainting
            # Using inpainting with mask to preserve face and only change clothing
            generated_image = await self.vast_ai_client.generate_img2img(
                image=image,
                mask=mask,
                prompt=STATIC_PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                denoising_strength=0.32,  # Fixed parameter
                steps=25,                  # Fixed parameter
                cfg_scale=7,               # Fixed parameter
                sampler_name="DPM++ 2M Karras",  # Fixed sampler
                width=512,                 # Fixed width
                height=768,                # Fixed height
                inpainting_fill=1,         # Inpainting fill mode
                inpaint_full_res=True,     # Full resolution inpainting
                inpaint_full_res_padding=32,  # Padding for full res
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
    
    def generate_clothing_mask(self, image: Image.Image) -> Image.Image:
        """
        Generate a mask for inpainting where only clothing area can be changed.
        
        Mask rules (VERY IMPORTANT):
        - White (255) = area SD CAN change (clothing)
        - Black (0) = area SD MUST NOT change (face, hair, body)
        
        If the face is white even a little → SD will change it.
        
        Args:
            image: Input PIL Image
        
        Returns:
            PIL Image mask (white = inpaint/clothing area, black = preserve)
        """
        from PIL import ImageDraw, ImageFilter
        
        width, height = image.size
        
        # Create mask: start with all black (preserve everything)
        mask = Image.new("L", (width, height), 0)  # 0 = black = preserve
        
        # Define clothing area (torso/body region)
        # IMPORTANT: Keep face area completely black (0) to preserve it
        # Face is typically in upper 30% of image
        # Clothing is typically in middle 30-70% of image height
        
        # Face protection zone (top 30% - keep black)
        face_bottom = int(height * 0.30)
        
        # Clothing area (middle section - make white for inpainting)
        clothing_top = int(height * 0.25)  # Start below face
        clothing_bottom = int(height * 0.75)  # End before legs
        clothing_left = int(width * 0.15)  # Margin on left
        clothing_right = int(width * 0.85)  # Margin on right
        
        # Draw white rectangle for clothing area only
        # This is the area SD can change
        draw = ImageDraw.Draw(mask)
        draw.rectangle(
            [(clothing_left, clothing_top), (clothing_right, clothing_bottom)],
            fill=255  # White = SD can change this area
        )
        
        # Apply Gaussian blur for soft edges (better blending)
        # This helps with smoother transitions
        mask = mask.filter(ImageFilter.GaussianBlur(radius=15))
        
        logger.info(f"Generated mask: clothing area ({clothing_left},{clothing_top}) to ({clothing_right},{clothing_bottom})")
        logger.info(f"Face area (top {face_bottom}px) is protected (black)")
        
        return mask
    
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
