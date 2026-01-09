"""
Image Processor for Virtual Try-On
Handles image processing and calls Vast.ai Stable Diffusion API for clothing generation.
"""

import io
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging
import uuid
from datetime import datetime

from vast_ai_client import VastAIClient
from config import (
    CONTROLNET_ENABLED, CONTROLNET_MODULE, CONTROLNET_MODEL, CONTROLNET_WEIGHT,
    CONTROLNET_GUIDANCE_START, CONTROLNET_GUIDANCE_END, CONTROLNET_CONTROL_MODE
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Static prompt for clothing replacement
# Hard-force pure white color with literal color command
STATIC_PROMPT = "plain solid pure white shirt, uniform white color, no shading, no gray, no patterns, no logos, pure white fabric"

# Negative prompt to avoid unwanted changes and enforce white color
NEGATIVE_PROMPT = "gray, off-white, shadows, wrinkles, texture transfer, skin, face, neck, hair, different face, face change, distorted face, new person, different person, changed identity, altered face, body deformation, extra limbs, bad anatomy, blur, low quality, face modification, patterns, stripes, designs, logos"




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
            
            # Generate mask using ControlNet segmentation
            # Step 1: Get segmentation map from ControlNet (detects body parts)
            logger.info("Getting segmentation map from ControlNet...")
            try:
                segmentation_map = await self.vast_ai_client.get_segmentation_map(image)
                
                # Step 2: Create mask from segmentation (clothing = white, everything else = black)
                logger.info("Creating mask from segmentation map...")
                mask = self.create_mask_from_segmentation(segmentation_map, image)
            except Exception as e:
                logger.warning(f"Segmentation failed: {e}. Falling back to rectangle mask.")
                # Fallback to rectangle mask if segmentation fails
                mask = self.generate_clothing_mask(image)
            
            # Generate image using Vast.ai img2img API with inpainting + ControlNet
            # Using inpainting with mask + ControlNet OpenPose to preserve face, pose, and only change clothing
            generated_image = await self.vast_ai_client.generate_img2img(
                image=image,
                mask=mask,
                prompt=STATIC_PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                denoising_strength=0.40,  # Moderate: enough for clothes, preserves face (0.35-0.45 range, max 0.45)
                steps=24,                  # Moderate steps (20-28 range)
                cfg_scale=5.0,             # Reduced model authority (4.5-6 range) to prevent face rewriting
                sampler_name="DPM++ 2M Karras",  # Fixed sampler
                width=512,                 # Fixed width
                height=768,                # Fixed height
                inpainting_fill=1,         # Inpainting fill mode
                inpaint_full_res=True,     # Preserves face details
                inpaint_full_res_padding=32,  # Smooth cloth boundaries
                controlnet_enabled=CONTROLNET_ENABLED,
                controlnet_model=CONTROLNET_MODEL if CONTROLNET_ENABLED else "control_sd15_openpose",
                controlnet_module=CONTROLNET_MODULE if CONTROLNET_ENABLED else "openpose",
                controlnet_weight=CONTROLNET_WEIGHT if CONTROLNET_ENABLED else 1.0,
                controlnet_guidance_start=CONTROLNET_GUIDANCE_START if CONTROLNET_ENABLED else 0.0,
                controlnet_guidance_end=CONTROLNET_GUIDANCE_END if CONTROLNET_ENABLED else 0.9,
                controlnet_control_mode=CONTROLNET_CONTROL_MODE if CONTROLNET_ENABLED else "Balanced"
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
    
    def create_mask_from_segmentation(self, segmentation_map: Image.Image, original_image: Image.Image) -> Image.Image:
        """
        Create inpainting mask from segmentation map.
        Converts clothing/shirt areas to white, everything else to black.
        
        Args:
            segmentation_map: Segmentation map from ControlNet (colored body part labels)
            original_image: Original input image (for size matching)
        
        Returns:
            PIL Image mask (white = clothing area, black = everything else)
        """
        import numpy as np
        from PIL import ImageFilter
        
        # Resize segmentation to match original image size
        width, height = original_image.size
        segmentation = segmentation_map.resize((width, height), Image.Resampling.LANCZOS)
        
        # Convert to numpy array for processing
        seg_array = np.array(segmentation)
        
        # Segmentation typically uses specific colors for different body parts
        # Common segmentation colors (may vary by model):
        # - Shirt/torso: Usually red/pink tones or specific color ranges
        # - Face: Usually skin tones
        # - Hair: Usually dark colors
        # - Background: Usually blue/green
        
        # Create mask: start with all black
        mask_array = np.zeros((height, width), dtype=np.uint8)
        
        # Convert segmentation to HSV for better color detection
        seg_hsv = Image.fromarray(seg_array).convert('HSV')
        hsv_array = np.array(seg_hsv)
        
        # Detect clothing/shirt areas
        # Shirt areas in segmentation are typically:
        # - Red/pink/magenta tones (H: 0-20 or 160-180)
        # - Medium to high saturation
        # - Medium to high value
        
        h, s, v = hsv_array[:, :, 0], hsv_array[:, :, 1], hsv_array[:, :, 2]
        
        # Define clothing color ranges in HSV
        # Red/pink/magenta for shirt (H: 0-20 or 160-180)
        clothing_mask_h = ((h >= 0) & (h <= 20)) | ((h >= 160) & (h <= 180))
        clothing_mask_s = (s >= 50) & (s <= 255)  # Medium to high saturation
        clothing_mask_v = (v >= 50) & (v <= 255)  # Medium to high value
        
        # Combine conditions for clothing detection
        clothing_mask = clothing_mask_h & clothing_mask_s & clothing_mask_v
        
        # Also check RGB directly for common shirt colors
        rgb_array = np.array(segmentation)
        r, g, b = rgb_array[:, :, 0], rgb_array[:, :, 1], rgb_array[:, :, 2]
        
        # Shirt areas often have: high red, medium blue/green
        # Or: pink/magenta tones
        rgb_clothing = (
            ((r > 150) & (r > g) & (r > b)) |  # Red/pink dominant
            ((r > 100) & (g < 100) & (b < 100)) |  # Red tones
            ((r > 120) & (g > 80) & (b > 120))  # Pink/magenta
        )
        
        # Combine HSV and RGB detection
        final_clothing_mask = clothing_mask | rgb_clothing
        
        # Set clothing areas to white (255)
        mask_array[final_clothing_mask] = 255
        
        # Clean up: remove any white pixels near face/neck/shoulder area
        # Lock face completely - mask must not touch neck, chin, jawline, or shoulders
        # Leave clear gap between bottom of face and top of shirt
        # Protect top 50% to ensure face, neck, chin, jawline, and shoulders are safe
        face_protection_zone = int(height * 0.50)
        mask_array[:face_protection_zone, :] = 0  # Force pure black in face/neck/shoulder area
        
        # Clean edges: remove small white pixels that might be face/neck
        # Use morphological operations to clean up
        try:
            from scipy import ndimage
            # Remove small isolated white pixels
            mask_array = ndimage.binary_opening(mask_array > 127, structure=np.ones((3, 3))).astype(np.uint8) * 255
            # Fill small holes
            mask_array = ndimage.binary_closing(mask_array > 127, structure=np.ones((5, 5))).astype(np.uint8) * 255
        except ImportError:
            # If scipy not available, use PIL filters
            logger.warning("scipy not available, using PIL filters for mask cleaning")
            mask_pil = Image.fromarray(mask_array, mode='L')
            # Apply filters to clean edges
            mask_pil = mask_pil.filter(ImageFilter.MinFilter(3))  # Remove small white pixels
            mask_pil = mask_pil.filter(ImageFilter.MaxFilter(5))   # Fill small holes
            mask_array = np.array(mask_pil)
        
        # Ensure face/neck/shoulder area is completely black (double protection)
        mask_array[:face_protection_zone, :] = 0
        
        # Convert to pure black/white (no gray pixels) - STRICT mask
        # Threshold: anything > 127 becomes 255 (white), <= 127 becomes 0 (black)
        # This ensures pure white (255) for shirt, pure black (0) for everything else
        mask_array = np.where(mask_array > 127, 255, 0).astype(np.uint8)
        
        # Ensure face/neck/shoulder area is pure black (triple protection)
        # Even a few gray pixels lets the model "rewrite" the face
        mask_array[:face_protection_zone, :] = 0
        
        # Convert back to PIL Image (NO BLUR, NO SOFT EDGES - strict mask)
        mask = Image.fromarray(mask_array, mode='L')
        
        # NO BLUR, NO SOFT EDGES - keep mask strict (pure black/white only)
        # Shirt area must be pure white (255), everything else pure black (0)
        # This ensures SD only changes the exact clothing area, no gray pixels
        
        logger.info(f"Created mask from segmentation: {np.sum(mask_array > 127)} white pixels")
        logger.info(f"Face area (top {face_protection_zone}px) is protected (black)")
        
        return mask
    
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
        # IMPORTANT: Keep face and head area completely black (0) to preserve person identity
        # Mask covers ONLY body/clothing area (white) - face, head, hair, background are black (preserved)
        # ControlNet will preserve body structure and person identity
        
        # Face and head protection zone (top 35% - keep black to preserve person identity)
        # Increased protection to ensure face/head/hair are never touched
        face_bottom = int(height * 0.35)
        
        # Clothing/body area ONLY (make white for inpainting)
        # Only the torso/body clothing area - face and head are protected
        clothing_top = int(height * 0.30)  # Start below face/head area
        clothing_bottom = int(height * 0.80)  # End before lower body/legs
        clothing_left = int(width * 0.12)  # Slight margins
        clothing_right = int(width * 0.88)  # Slight margins
        
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
