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
            # Step 1: Detect full person and clothing using ControlNet
            logger.info("Detecting person and clothing using ControlNet...")
            detections = await self.vast_ai_client.detect_person_and_clothing(image)
            face_detection = detections.get('face')
            segmentation_map = detections.get('segmentation')
            
            if face_detection is not None:
                logger.info("✅ Face detection successful - using detected face area for protection")
            else:
                logger.warning("⚠️  Face detection failed - will use fallback protection")
            
            if segmentation_map is not None:
                logger.info("✅ Person/body segmentation successful - detecting clothing regions")
            else:
                logger.warning("⚠️  Segmentation failed - will use geometric mask")
            
            # Step 2: Generate clothing mask based on detected clothing
            # Only mask clothing that actually exists in the image
            logger.info("Generating clothing mask from detected clothing...")
            mask, clothing_items = self.generate_clothing_mask_from_segmentation(
                image, 
                face_detection, 
                segmentation_map
            )
            
            # Step 3: Update prompt based on detected clothing
            # Only mention clothing that exists (e.g., if no pants, don't mention pants)
            prompt, negative_prompt = self.build_prompt_from_clothing(clothing_items)
            
            # Verify mask is correct: should have white pixels in clothing region only
            mask_array = np.array(mask)
            white_pixels = np.sum(mask_array == 255)
            total_pixels = mask_array.size
            white_ratio = white_pixels / total_pixels
            logger.info(f"Mask verification: {white_pixels} white pixels ({white_ratio*100:.1f}% of image)")
            
            if white_ratio < 0.05 or white_ratio > 0.40:
                logger.warning(f"Mask white ratio ({white_ratio*100:.1f}%) seems unusual. Expected 10-30% for clothing region.")
            
            # Final verification: ensure mask has sufficient white pixels for shirt area
            final_white = np.sum(mask_array == 255)
            final_white_ratio = final_white / mask_array.size
            logger.info(f"=== MASK VERIFICATION ===")
            logger.info(f"White pixels: {final_white} ({final_white_ratio*100:.2f}% of image)")
            logger.info(f"Black pixels: {mask_array.size - final_white} ({(1-final_white_ratio)*100:.2f}% of image)")
            # Count face area white pixels (should be 0)
            face_white_pixels = np.sum(mask_array[:int(mask_array.shape[0] * 0.3), :] == 255)
            logger.info(f"Face area (top 30%): {face_white_pixels} white pixels (should be 0)")
            
            if final_white_ratio < 0.10:
                logger.warning(f"⚠️  Mask white ratio ({final_white_ratio*100:.1f}%) is very low. Shirt might not change.")
                logger.warning(f"   Consider increasing mask size if shirt doesn't change.")
            elif final_white_ratio > 0.50:
                logger.warning(f"⚠️  Mask white ratio ({final_white_ratio*100:.1f}%) is very high. Background might change.")
            else:
                logger.info(f"✅ Mask white ratio ({final_white_ratio*100:.1f}%) is good for shirt replacement.")
            
            # Save mask for debugging (optional - can be removed in production)
            try:
                mask_debug_path = self.temp_dir / f"mask_debug_{uuid.uuid4().hex[:8]}.png"
                mask.save(mask_debug_path)
                logger.info(f"Mask saved for debugging: {mask_debug_path}")
            except Exception as e:
                logger.warning(f"Could not save debug mask: {e}")
            
            # Generate image using Vast.ai img2img INPAINT API with inpainting
            # Use balanced denoising + natural blending parameters for seamless clothing replacement
            generated_image = await self.vast_ai_client.generate_img2img(
                image=image,
                mask=mask,
                prompt=prompt,
                negative_prompt=negative_prompt,
                denoising_strength=0.35,  # Balanced denoising (0.30-0.35): ensures shirt changes while preserving face
                steps=30,  # More steps for better quality and blending
                cfg_scale=6,  # Balanced CFG for natural results
                sampler_name="DPM++ SDE",
                width=1024,
                height=1024,
                # ControlNet Unit 0: Pose lock (preserves full body pose)
                controlnet_pose_enabled=True,
                controlnet_pose_module="openpose_full",  # Full body pose detection
                controlnet_pose_model="controlnet-openpose-sdxl",  # OpenPose model
                controlnet_pose_weight=1.0,  # Weight 1.0 for strong pose preservation
                controlnet_pose_control_mode="ControlNet is more important",  # Strong control to preserve person
                # ControlNet Unit 1: Inpaint guidance (helps with clothing replacement)
                controlnet_inpaint_enabled=True,
                controlnet_inpaint_model="controlnet-inpaint-sdxl",  # Inpaint model for guidance
                controlnet_inpaint_weight=1.0,  # Weight 1.0 for inpainting guidance
                controlnet_pixel_perfect=True  # Pixel perfect mode
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
            error_msg = str(e)
            logger.error(f"❌ Error processing image: {error_msg}")
            logger.error(f"Error type: {type(e).__name__}")
            
            # Provide more helpful error messages
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                error_msg = f"Image generation timed out. The API may be overloaded or the image is too complex. Please try again."
            elif "connection" in error_msg.lower() or "connect" in error_msg.lower():
                error_msg = f"Cannot connect to image generation API. Please check if the server is running."
            elif "no images" in error_msg.lower() or "images" in error_msg.lower():
                error_msg = f"API did not return a generated image. The generation may have failed."
            
            return {
                "success": False,
                "error": error_msg
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
    
    def create_face_mask_from_detection(self, face_detection: Image.Image, original_image: Image.Image) -> Image.Image:
        """
        Create a face protection mask from ControlNet face detection.
        
        Args:
            face_detection: Face detection map from ControlNet
            original_image: Original input image (for size matching)
        
        Returns:
            PIL Image mask (white = face area to protect, black = everything else)
        """
        import numpy as np
        
        width, height = original_image.size
        
        # Resize detection to match original image size
        detection = face_detection.resize((width, height), Image.Resampling.LANCZOS)
        detection_array = np.array(detection)
        
        # Create face mask: white where face is detected, black elsewhere
        # Face detection typically shows face as bright/colored areas
        # Convert to grayscale if needed
        if len(detection_array.shape) == 3:
            # RGB image - convert to grayscale
            detection_gray = np.mean(detection_array, axis=2).astype(np.uint8)
        else:
            detection_gray = detection_array
        
        # Threshold: bright areas are likely face
        # Face detection maps usually have high values where face is detected
        threshold = np.percentile(detection_gray, 70)  # Top 30% brightest = face area
        face_mask_array = (detection_gray > threshold).astype(np.uint8) * 255
        
        # Dilate the mask slightly to ensure full face coverage
        try:
            from scipy import ndimage
            face_mask_array = ndimage.binary_dilation(face_mask_array > 127, structure=np.ones((5, 5))).astype(np.uint8) * 255
        except ImportError:
            # If scipy not available, use PIL filters
            from PIL import ImageFilter
            face_mask_pil = Image.fromarray(face_mask_array, mode='L')
            face_mask_pil = face_mask_pil.filter(ImageFilter.MaxFilter(5))  # Dilate
            face_mask_array = np.array(face_mask_pil)
        
        logger.info(f"Face mask created: {np.sum(face_mask_array == 255)} white pixels ({np.sum(face_mask_array == 255)/face_mask_array.size*100:.1f}% of image)")
        
        return Image.fromarray(face_mask_array, mode='L')
    
    def detect_clothing_from_segmentation(self, segmentation_map: Image.Image, original_image: Image.Image) -> Dict[str, bool]:
        """
        Detect what clothing items exist in the image from segmentation map.
        
        Args:
            segmentation_map: Segmentation map from ControlNet
            original_image: Original input image
        
        Returns:
            Dictionary with clothing items and whether they exist:
            {'has_shirt': bool, 'has_pants': bool, 'has_dress': bool}
        """
        width, height = original_image.size
        segmentation = segmentation_map.resize((width, height), Image.Resampling.LANCZOS)
        seg_array = np.array(segmentation)
        
        # Convert to HSV for better color detection
        if len(seg_array.shape) == 3:
            seg_hsv = Image.fromarray(seg_array).convert('HSV')
            hsv_array = np.array(seg_hsv)
            h, s, v = hsv_array[:, :, 0], hsv_array[:, :, 1], hsv_array[:, :, 2]
        else:
            h, s, v = seg_array, seg_array, seg_array
        
        # Detect clothing regions based on segmentation colors
        # Segmentation typically uses specific colors for different body parts:
        # - Shirt/torso: Usually red/pink/magenta tones
        # - Pants/legs: Usually blue/cyan tones
        # - Face/head: Usually yellow/orange tones
        
        # Shirt detection: look for red/pink/magenta in middle-upper region (torso)
        torso_region = seg_array[int(height*0.2):int(height*0.7), :]
        if len(torso_region.shape) == 3:
            torso_hsv = Image.fromarray(torso_region).convert('HSV')
            torso_h = np.array(torso_hsv)[:, :, 0]
        else:
            torso_h = torso_region
        
        # Red/pink/magenta tones (H: 0-20 or 160-180)
        shirt_pixels = np.sum(((torso_h >= 0) & (torso_h <= 20)) | ((torso_h >= 160) & (torso_h <= 180)))
        has_shirt = shirt_pixels > (torso_region.size * 0.1)  # At least 10% of torso region
        
        # Pants detection: look for blue/cyan in lower region (legs)
        legs_region = seg_array[int(height*0.6):int(height*0.95), :]
        if len(legs_region.shape) == 3:
            legs_hsv = Image.fromarray(legs_region).convert('HSV')
            legs_h = np.array(legs_hsv)[:, :, 0]
        else:
            legs_h = legs_region
        
        # Blue/cyan tones (H: 100-130)
        pants_pixels = np.sum((legs_h >= 100) & (legs_h <= 130))
        has_pants = pants_pixels > (legs_region.size * 0.1)  # At least 10% of legs region
        
        # Also check RGB directly for common segmentation colors
        if len(seg_array.shape) == 3:
            r, g, b = seg_array[:, :, 0], seg_array[:, :, 1], seg_array[:, :, 2]
            # Shirt: high red, medium green/blue
            shirt_rgb = np.sum((r > 150) & (r > g) & (r > b))
            if shirt_rgb > (seg_array.shape[0] * seg_array.shape[1] * 0.05):
                has_shirt = True
            # Pants: high blue, medium red/green
            pants_rgb = np.sum((b > 150) & (b > r) & (b > g))
            if pants_rgb > (seg_array.shape[0] * seg_array.shape[1] * 0.05):
                has_pants = True
        
        logger.info(f"Clothing detection: has_shirt={has_shirt}, has_pants={has_pants}")
        
        return {
            'has_shirt': has_shirt,
            'has_pants': has_pants,
            'has_dress': False  # Can be extended later
        }
    
    def build_prompt_from_clothing(self, clothing_items: Dict[str, bool]) -> Tuple[str, str]:
        """
        Build prompt based on detected clothing items.
        Only mention clothing that exists.
        
        Args:
            clothing_items: Dictionary with clothing detection results
        
        Returns:
            Tuple of (prompt, negative_prompt)
        """
        prompt_parts = []
        negative_parts = [
            "different person", "new person", "face change", "face modification", 
            "altered face", "different face", "mannequin", "product photo", 
            "studio shot", "floating clothes", "flat lay", "folded shirt", 
            "catalog image", "jacket", "hoodie", "coat", "logo", "pattern", 
            "distorted face", "face deformation", "pasted", "overlaid", 
            "digital overlay", "sharp edges", "visible seams"
        ]
        
        # Add clothing-specific prompts based on what exists
        if clothing_items.get('has_shirt', True):  # Default to True if not detected
            prompt_parts.extend([
                "white cotton shirt", "white button-up shirt", 
                "white long-sleeved shirt", "white shirt", 
                "clean white fabric", "white sleeves",
                "visible white shirt", "clear white shirt", 
                "fully visible white clothing", "white shirt clearly visible",
                "white shirt not hidden", "white shirt unobstructed"
            ])
            negative_parts.extend([
                "gray shirt", "black shirt", "colored shirt", 
                "dark shirt", "blue shirt", "red shirt", 
                "charcoal shirt", "grey shirt", "dark grey shirt",
                "hidden shirt", "obscured shirt", "covered shirt",
                "translucent object", "semi-transparent", "foggy",
                "blurred clothing", "unclear shirt", "invisible shirt"
            ])
        
        if clothing_items.get('has_pants', False):  # Only if pants detected
            prompt_parts.extend([
                "white pants", "white trousers", "white clothing"
            ])
            negative_parts.extend([
                "dark pants", "black pants", "gray pants", 
                "jeans", "colored pants"
            ])
        
        # Common parts
        prompt_parts.extend([
            "natural clothing texture", "seamlessly integrated", 
            "natural clothing replacement", "realistic white clothing",
            "same person", "same pose", "same background", 
            "natural fabric folds", "professional white clothing",
            "clearly visible clothing", "sharp white shirt", 
            "crisp white fabric", "well-lit white shirt"
        ])
        
        prompt = ", ".join(prompt_parts)
        negative_prompt = ", ".join(negative_parts)
        
        logger.info(f"Generated prompt: {prompt[:100]}...")
        logger.info(f"Clothing items in prompt: shirt={clothing_items.get('has_shirt')}, pants={clothing_items.get('has_pants')}")
        
        return prompt, negative_prompt
    
    def generate_clothing_mask_from_segmentation(
        self, 
        image: Image.Image, 
        face_detection: Optional[Image.Image] = None,
        segmentation_map: Optional[Image.Image] = None
    ) -> Tuple[Image.Image, Dict[str, bool]]:
        """
        Generate clothing mask based on detected clothing from segmentation.
        Only masks clothing that actually exists in the image.
        
        Args:
            image: Input PIL Image
            face_detection: Face detection map from ControlNet (optional)
            segmentation_map: Segmentation map from ControlNet (optional)
        
        Returns:
            Tuple of (mask Image, clothing_items dict)
        """
        from PIL import ImageDraw
        
        width, height = image.size
        
        # Detect what clothing exists
        clothing_items = {'has_shirt': True, 'has_pants': False, 'has_dress': False}
        if segmentation_map is not None:
            try:
                clothing_items = self.detect_clothing_from_segmentation(segmentation_map, image)
            except Exception as e:
                logger.warning(f"Clothing detection from segmentation failed: {e}, using defaults")
        
        # Create mask: start with ALL BLACK (preserve everything)
        mask = Image.new("L", (width, height), 0)  # 0 = black = preserve
        
        # Create face protection mask
        face_mask = None
        if face_detection is not None:
            try:
                face_mask = self.create_face_mask_from_detection(face_detection, image)
                logger.info("Using ControlNet face detection for face protection")
            except Exception as e:
                logger.warning(f"Failed to create face mask: {e}, using fallback")
                face_mask = None
        
        face_protection_fallback = int(height * 0.20) if face_mask is None else None
        
        draw = ImageDraw.Draw(mask)
        
        # Only mask clothing that exists
        if clothing_items.get('has_shirt', True):
            # Shirt region
            shirt_top = int(height * 0.15)
            shirt_bottom = int(height * 0.78)
            chest_left = int(width * 0.08)
            chest_right = int(width * 0.92)
            shoulder_left = int(width * 0.02)
            shoulder_right = int(width * 0.98)
            
            # Main torso/chest
            draw.rectangle(
                [(chest_left, shirt_top), (chest_right, shirt_bottom)],
                fill=255  # White = change shirt
            )
            
            # Arms
            left_arm_bottom = int(height * 0.75)
            draw.rectangle(
                [(shoulder_left, shirt_top), (chest_left, left_arm_bottom)],
                fill=255  # White = change sleeves
            )
            draw.rectangle(
                [(chest_right, shirt_top), (shoulder_right, left_arm_bottom)],
                fill=255  # White = change sleeves
            )
            logger.info("Masked shirt area")
        
        if clothing_items.get('has_pants', False):
            # Pants region - only if pants detected
            pants_top = int(height * 0.70)
            pants_bottom = int(height * 0.92)
            pants_left = int(width * 0.20)
            pants_right = int(width * 0.80)
            
            draw.rectangle(
                [(pants_left, pants_top), (pants_right, pants_bottom)],
                fill=255  # White = change pants
            )
            logger.info("Masked pants area")
        else:
            logger.info("No pants detected - skipping pants mask")
        
        # Apply face protection
        if face_mask is not None:
            mask_array = np.array(mask)
            face_mask_array = np.array(face_mask)
            mask_array[face_mask_array > 127] = 0
            mask = Image.fromarray(mask_array, mode='L')
            logger.info("Applied ControlNet face detection mask")
        else:
            draw.rectangle(
                [(0, 0), (width, face_protection_fallback)],
                fill=0  # Black = preserve face
            )
            logger.info(f"Using fallback face protection (top {face_protection_fallback*100/height:.1f}%)")
        
        # Re-draw clothing after face protection
        draw = ImageDraw.Draw(mask)
        if clothing_items.get('has_shirt', True):
            draw.rectangle([(chest_left, shirt_top), (chest_right, shirt_bottom)], fill=255)
            draw.rectangle([(shoulder_left, shirt_top), (chest_left, left_arm_bottom)], fill=255)
            draw.rectangle([(chest_right, shirt_top), (shoulder_right, left_arm_bottom)], fill=255)
        
        if clothing_items.get('has_pants', False):
            draw.rectangle([(pants_left, pants_top), (pants_right, pants_bottom)], fill=255)
        
        # Re-apply face protection
        if face_mask is not None:
            mask_array = np.array(mask)
            face_mask_array = np.array(face_mask)
            mask_array[face_mask_array > 127] = 0
            mask = Image.fromarray(mask_array, mode='L')
        
        # Final cleanup
        mask_array = np.array(mask)
        mask_array = np.where(mask_array > 127, 255, 0).astype(np.uint8)
        mask = Image.fromarray(mask_array, mode='L')
        
        return mask, clothing_items
    
    def generate_clothing_mask(self, image: Image.Image, face_detection: Optional[Image.Image] = None) -> Image.Image:
        """
        Generate a precise mask for inpainting where ALL CLOTHES can be changed.
        
        CRITICAL MASK RULES:
        - White (255) = ALL clothing: Shirt (full), Pants/Trousers, Upper arms, Shoulder fabric
        - Black (0) = everything else: Face, Neck, Background, Shoes, Feet, Hands
        
        If mask is:
        - Too large → background changes
        - Too small → clothes don't change
        - White on face → FACE CHANGES (THIS IS THE BUG!)
        - Missing → model ignores person entirely
        
        Args:
            image: Input PIL Image
        
        Returns:
            PIL Image mask (white = ALL clothing area, black = preserve everything else)
        """
        from PIL import ImageDraw
        
        width, height = image.size
        
        # Create mask: start with ALL BLACK (preserve everything)
        # This is critical - we only mark clothing area as white
        mask = Image.new("L", (width, height), 0)  # 0 = black = preserve
        
        # Create face protection mask from ControlNet detection if available
        face_mask = None
        if face_detection is not None:
            try:
                face_mask = self.create_face_mask_from_detection(face_detection, image)
                logger.info("Using ControlNet face detection for face protection")
            except Exception as e:
                logger.warning(f"Failed to create face mask from detection: {e}, using fallback")
                face_mask = None
        
        # Fallback: if face detection failed, use conservative top 20% protection
        # This is much smaller than before to allow shirt collar to be changed
        face_protection_fallback = int(height * 0.20) if face_mask is None else None
        
        # CLOTHING REGION: Cover ALL clothes (shirt + pants)
        # Shirt region: start from very top to catch full shirt including collar
        # We'll protect face area separately using detected face mask
        shirt_top = int(height * 0.15)  # Start high to catch shirt collar (will be protected by face mask)
        shirt_bottom = int(height * 0.78)  # Extend lower to cover full shirt including untucked part
        
        # Pants/Trousers region: from waist to just above shoes
        pants_top = int(height * 0.70)  # Start at waist (overlaps with shirt bottom)
        pants_bottom = int(height * 0.92)  # End just above shoes/feet
        
        # Chest/shirt area (center torso) - EXPANDED width for full shirt coverage
        chest_left = int(width * 0.08)  # Left edge of shirt (wider - was 0.12)
        chest_right = int(width * 0.92)  # Right edge of shirt (wider - was 0.88)
        
        # Full arms and shoulders - cover ENTIRE sleeves including rolled-up parts
        shoulder_left = int(width * 0.02)  # Left shoulder/arm (wider - was 0.05)
        shoulder_right = int(width * 0.98)  # Right shoulder/arm (wider - was 0.95)
        
        # Pants width (narrower than shirt, centered)
        pants_left = int(width * 0.20)  # Left edge of pants
        pants_right = int(width * 0.80)  # Right edge of pants
        
        # Draw clothing regions: shirt + pants
        draw = ImageDraw.Draw(mask)
        
        # ===== SHIRT REGION - FULL COVERAGE =====
        # Main torso/chest rectangle (WHITE = change this area)
        # Make it wider and taller to ensure full shirt coverage
        draw.rectangle(
            [(chest_left, shirt_top), (chest_right, shirt_bottom)],
            fill=255  # White = SD can change this area (FULL shirt including lower part)
        )
        
        # Left arm and shoulder - FULL sleeve coverage (including rolled-up sleeves)
        # Extend arms to cover entire sleeve area from shoulder to elbow and below
        left_arm_top = shirt_top
        left_arm_bottom = int(height * 0.75)  # Arm extends further down to cover rolled sleeves (was 0.70)
        draw.rectangle(
            [(shoulder_left, left_arm_top), (chest_left, left_arm_bottom)],
            fill=255  # White = FULL arm/shoulder area (entire shirt sleeve)
        )
        
        # Right arm and shoulder - FULL sleeve coverage (including rolled-up sleeves)
        right_arm_top = shirt_top
        right_arm_bottom = int(height * 0.75)  # Arm extends further down to cover rolled sleeves (was 0.70)
        draw.rectangle(
            [(chest_right, right_arm_top), (shoulder_right, right_arm_bottom)],
            fill=255  # White = FULL arm/shoulder area (entire shirt sleeve)
        )
        
        # ===== PANTS/TROUSERS REGION =====
        # Pants rectangle (WHITE = change this area)
        draw.rectangle(
            [(pants_left, pants_top), (pants_right, pants_bottom)],
            fill=255  # White = SD can change this area (pants/trousers)
        )
        
        # CRITICAL: Apply face protection mask
        # This MUST be done AFTER drawing shirt/pants to ensure face protection
        if face_mask is not None:
            # Use detected face mask to protect face area precisely
            face_mask_array = np.array(face_mask)
            # Invert: face mask is white where face is, we want black (protect)
            # So set mask to black where face_mask is white
            mask_array = np.array(mask)
            mask_array[face_mask_array > 127] = 0  # Protect face area (set to black)
            mask = Image.fromarray(mask_array, mode='L')
            logger.info("Applied ControlNet face detection mask for face protection")
        else:
            # Fallback: use percentage-based protection (conservative)
            draw.rectangle(
                [(0, 0), (width, face_protection_fallback)],
                fill=0  # Black = preserve face/head area (DO NOT CHANGE)
            )
            logger.info(f"Using fallback face protection (top {face_protection_fallback*100/height:.1f}%)")
        
        # Re-draw shirt area AFTER face protection to ensure shirt mask is not lost
        # This ensures shirt area (outside face) remains white, including collar area
        draw = ImageDraw.Draw(mask)
        draw.rectangle(
            [(chest_left, shirt_top), (chest_right, shirt_bottom)],
            fill=255  # White = SD can change this area (FULL shirt including collar, except face)
        )
        draw.rectangle(
            [(shoulder_left, left_arm_top), (chest_left, left_arm_bottom)],
            fill=255  # White = FULL arm/shoulder area
        )
        draw.rectangle(
            [(chest_right, right_arm_top), (shoulder_right, right_arm_bottom)],
            fill=255  # White = FULL arm/shoulder area
        )
        
        # Re-apply face protection after re-drawing shirt
        if face_mask is not None:
            mask_array = np.array(mask)
            face_mask_array = np.array(face_mask)
            mask_array[face_mask_array > 127] = 0  # Protect face area again
            mask = Image.fromarray(mask_array, mode='L')
        
        # Ensure bottom area (shoes, feet, background) is black
        draw.rectangle(
            [(0, pants_bottom), (width, height)],
            fill=0  # Black = preserve shoes, feet, background
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
        
        # TRIPLE-CHECK: Force face area to be black using face mask or fallback
        mask_array = np.array(mask)
        if face_mask is not None:
            face_mask_array = np.array(face_mask)
            mask_array[face_mask_array > 127] = 0  # Protect detected face area
        else:
            # Fallback protection
            mask_array[:face_protection_fallback, :] = 0
        
        # Verify mask correctness
        white_pixels = np.sum(mask_array == 255)
        total_pixels = mask_array.size
        
        # Check face area white pixels
        if face_mask is not None:
            face_mask_array = np.array(face_mask)
            face_area_pixels = np.sum((mask_array == 255) & (face_mask_array > 127))
            if face_area_pixels > 0:
                logger.error(f"CRITICAL ERROR: {face_area_pixels} white pixels found in detected face area! Forcing to black...")
                mask_array[face_mask_array > 127] = 0
            face_protection_info = f"ControlNet detected face area"
        else:
            face_area_pixels = np.sum(mask_array[:face_protection_fallback, :] == 255)
            if face_area_pixels > 0:
                logger.error(f"CRITICAL ERROR: {face_area_pixels} white pixels found in face area! Forcing to black...")
                mask_array[:face_protection_fallback, :] = 0
            face_protection_info = f"Fallback protection (top {face_protection_fallback*100/height:.1f}%)"
        
        logger.info(f"Generated clothing mask (shirt + pants):")
        logger.info(f"  - White pixels (clothing area): {white_pixels} ({white_pixels/total_pixels*100:.1f}%)")
        logger.info(f"  - Face protection: {face_protection_info}")
        logger.info(f"  - Mask includes: Shirt (FULL including collar), Pants/Trousers, Upper arms, Shoulder fabric")
        logger.info(f"  - Mask excludes: Face/head (detected or top 20%), Background, Shoes, Feet, Hands")
        logger.info(f"  - Shirt mask: {shirt_top*100/height:.1f}% to {shirt_bottom*100/height:.1f}% of image height")
        
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
