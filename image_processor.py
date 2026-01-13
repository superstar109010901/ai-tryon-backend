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
    CONTROLNET_GUIDANCE_START, CONTROLNET_GUIDANCE_END, CONTROLNET_CONTROL_MODE,
    BACKEND_URL
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Static prompt for clothing replacement
# Clothes-only prompt - focuses only on the clothing
STATIC_PROMPT = "plain white cotton shirt"

# Negative prompt - strong exclusion of face, hair, skin, body, identity
NEGATIVE_PROMPT = "face, hair, skin, body, identity, person, head, neck, different face, face change, distorted face, new person, different person, changed identity, altered face, body deformation, extra limbs, bad anatomy, blur, low quality, face modification, original shirt, original color, gray, black, texture, pattern, logo, shadows"




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
            
            # Resize image to match target dimensions (1024x1024)
            target_width = 1024
            target_height = 1024
            if image.size != (target_width, target_height):
                image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
                logger.info(f"Resized image to {image.size}")
            
            # Generate mask for inpainting
            # CRITICAL: Use reliable geometric mask instead of segmentation
            # Segmentation can misdetect face as clothing or vice versa
            # Geometric mask ensures: White = shirt area (torso), Black = everything else (face, neck, background)
            logger.info("Generating precise torso mask (chest, upper arms, shoulders)...")
            mask = self.generate_clothing_mask(image)
            
            # Verify mask is correct: should have white pixels in torso region only
            mask_array = np.array(mask)
            white_pixels = np.sum(mask_array == 255)
            total_pixels = mask_array.size
            white_ratio = white_pixels / total_pixels
            logger.info(f"Mask verification: {white_pixels} white pixels ({white_ratio*100:.1f}% of image)")
            
            if white_ratio < 0.05 or white_ratio > 0.40:
                logger.warning(f"Mask white ratio ({white_ratio*100:.1f}%) seems unusual. Expected 10-30% for torso region.")
            
            # Ensure top 45% is completely black (face/neck protection)
            height = mask_array.shape[0]
            face_protection = int(height * 0.45)
            mask_array[:face_protection, :] = 0
            mask = Image.fromarray(mask_array, mode='L')
            logger.info(f"Face/neck area (top {face_protection}px) forced to black")
            
            # Generate image using Vast.ai img2img INPAINT API with inpainting + ControlNet
            # Using exact payload structure: low denoise (0.4) + inpainting params + tightened ControlNet
            generated_image = await self.vast_ai_client.generate_img2img(
                image=image,
                mask=mask,
                prompt="plain white cotton shirt, clean fabric, natural fabric folds, realistic clothing texture",
                negative_prompt="different person, new person, face change, face modification, altered face, different face, mannequin, product photo, studio shot, floating clothes, flat lay, folded shirt, catalog image, jacket, hoodie, coat, logo, pattern, distorted face, face deformation",
                denoising_strength=0.4,  # Low denoise: freeze everything except torso
                steps=25,
                cfg_scale=5,
                sampler_name="DPM++ SDE",
                width=1024,
                height=1024,
                # ControlNet configuration - tightened for person preservation
                controlnet_enabled=True,
                controlnet_model="controlnet-inpaint-dreamer-sdxl",
                controlnet_module="none",
                controlnet_weight=1.2,  # Increased weight for stronger control
                controlnet_control_mode="ControlNet is more important",  # Tell SD: DO NOT IGNORE THE PERSON
                controlnet_pixel_perfect=True
            )
            
            # Save generated image temporarily
            filename = f"generated_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}.png"
            filepath = self.temp_dir / filename
            generated_image.save(filepath, "PNG")
            
            # Clean up old files (keep only last 10)
            self.cleanup_old_files(keep_count=10)
            
            logger.info(f"Image processed successfully: {filename}")
            
            # Return full backend URL for the image
            # Frontend will use this URL directly to download the image
            image_url = f"{BACKEND_URL}/download/{filename}"
            
            return {
                "success": True,
                "image_url": image_url,
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
        
        # CRITICAL: Verify mask is correct and active
        # If segmentation shows clothes as one color and background as another,
        # we may need to INVERT the mask
        # Check: if most of the image is white, mask is likely inverted
        white_pixel_ratio = np.sum(mask_array == 255) / mask_array.size
        if white_pixel_ratio > 0.7:  # If more than 70% is white, likely inverted
            logger.warning(f"Mask appears inverted (white_pixel_ratio={white_pixel_ratio:.2f}). Inverting mask...")
            mask_array = np.where(mask_array == 255, 0, 255).astype(np.uint8)
            # Re-protect face area after inversion
            mask_array[:face_protection_zone, :] = 0
            white_pixel_ratio = np.sum(mask_array == 255) / mask_array.size
            logger.info(f"After inversion: white_pixel_ratio={white_pixel_ratio:.2f}")
        
        # Ensure face/neck/shoulder area is pure black (triple protection)
        # Even a few gray pixels lets the model "rewrite" the face
        mask_array[:face_protection_zone, :] = 0
        
        # Final verification: ensure mask is active and dominant
        # Shirt area must be pure white (255), everything else pure black (0)
        # No transparency, no gray - force pure black/white
        mask_array = np.where(mask_array > 127, 255, 0).astype(np.uint8)
        
        # Log mask statistics for debugging
        white_pixels = np.sum(mask_array == 255)
        black_pixels = np.sum(mask_array == 0)
        total_pixels = mask_array.size
        logger.info(f"Mask verification: {white_pixels} white pixels ({white_pixels/total_pixels*100:.1f}%), {black_pixels} black pixels ({black_pixels/total_pixels*100:.1f}%)")
        
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
        Generate a precise mask for inpainting where only torso region can be changed.
        
        CRITICAL MASK RULES:
        - White (255) = shirt area ONLY: Chest, Upper arms, Shoulder fabric
        - Black (0) = everything else: Face, Neck, Background, Table, Hands
        
        If mask is:
        - Too large → background changes
        - Too small → shirt floats/disappears
        - White on face → FACE CHANGES (THIS IS THE BUG!)
        - Missing → model ignores person entirely
        
        Args:
            image: Input PIL Image
        
        Returns:
            PIL Image mask (white = torso/shirt area, black = preserve everything else)
        """
        from PIL import ImageDraw
        
        width, height = image.size
        
        # Create mask: start with ALL BLACK (preserve everything)
        # This is critical - we only mark shirt area as white
        mask = Image.new("L", (width, height), 0)  # 0 = black = preserve
        
        # CRITICAL: Face and neck protection zone (top 45% - MUST be black)
        # Even ONE white pixel in face area will cause face changes
        face_neck_bottom = int(height * 0.45)
        
        # Torso region (chest, upper arms, shoulders)
        # Start BELOW neck, end before lower body
        torso_top = int(height * 0.45)  # Start below neck (safe zone)
        torso_bottom = int(height * 0.80)  # End before lower body/waist
        
        # Chest area (center torso)
        chest_left = int(width * 0.18)  # Left edge of chest
        chest_right = int(width * 0.82)  # Right edge of chest
        
        # Upper arms and shoulders (wider than chest)
        # Shoulders extend beyond chest but stay within person bounds
        shoulder_left = int(width * 0.08)  # Left shoulder/arm
        shoulder_right = int(width * 0.92)  # Right shoulder/arm
        
        # Draw torso region: chest + upper arms + shoulders
        draw = ImageDraw.Draw(mask)
        
        # Main torso/chest rectangle (WHITE = change this area)
        draw.rectangle(
            [(chest_left, torso_top), (chest_right, torso_bottom)],
            fill=255  # White = SD can change this area (shirt only)
        )
        
        # Left upper arm and shoulder (WHITE = change this area)
        left_arm_top = torso_top
        left_arm_bottom = int(height * 0.65)  # Upper arm ends mid-torso
        draw.rectangle(
            [(shoulder_left, left_arm_top), (chest_left, left_arm_bottom)],
            fill=255  # White = upper arm/shoulder area (shirt sleeve)
        )
        
        # Right upper arm and shoulder (WHITE = change this area)
        right_arm_top = torso_top
        right_arm_bottom = int(height * 0.65)  # Upper arm ends mid-torso
        draw.rectangle(
            [(chest_right, right_arm_top), (shoulder_right, right_arm_bottom)],
            fill=255  # White = upper arm/shoulder area (shirt sleeve)
        )
        
        # CRITICAL: Force face/neck area to be COMPLETELY BLACK
        # This is the most important step - even a few white pixels will change the face
        draw.rectangle(
            [(0, 0), (width, face_neck_bottom)],
            fill=0  # Black = preserve face and neck (DO NOT CHANGE)
        )
        
        # Ensure bottom area (table, background, hands) is black
        draw.rectangle(
            [(0, torso_bottom), (width, height)],
            fill=0  # Black = preserve background/table/hands
        )
        
        # Ensure side margins (background) are black
        draw.rectangle(
            [(0, 0), (shoulder_left, height)],
            fill=0  # Black = preserve left background
        )
        draw.rectangle(
            [(shoulder_right, 0), (width, height)],
            fill=0  # Black = preserve right background
        )
        
        # Convert to pure black/white (no gray pixels)
        # This ensures strict mask: white = change, black = preserve
        mask_array = np.array(mask)
        mask_array = np.where(mask_array > 127, 255, 0).astype(np.uint8)
        
        # TRIPLE-CHECK: Force face area to be black (even if somehow white pixels got in)
        mask_array[:face_neck_bottom, :] = 0
        
        # Verify mask correctness
        white_pixels = np.sum(mask_array == 255)
        face_area_pixels = np.sum(mask_array[:face_neck_bottom, :] == 255)
        total_pixels = mask_array.size
        
        if face_area_pixels > 0:
            logger.error(f"CRITICAL ERROR: {face_area_pixels} white pixels found in face area! Forcing to black...")
            mask_array[:face_neck_bottom, :] = 0
        
        logger.info(f"Generated precise torso mask:")
        logger.info(f"  - White pixels (shirt area): {white_pixels} ({white_pixels/total_pixels*100:.1f}%)")
        logger.info(f"  - Face area white pixels: {np.sum(mask_array[:face_neck_bottom, :] == 255)} (should be 0)")
        logger.info(f"  - Mask includes: Chest, Upper arms, Shoulder fabric")
        logger.info(f"  - Mask excludes: Face (top {face_neck_bottom}px), Neck, Background, Table")
        
        mask = Image.fromarray(mask_array, mode='L')
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
