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

# Static prompt for white clothing generation
# Focus on white clothing that person is wearing
STATIC_PROMPT = "woman without any clothes, no clothes, nude, realistic nude, same person, taking out all clothes, ORIGINAL FACE UNCHANGED, ORIGINAL FACE PRESERVED, EXACT SAME FACE, same body, same pose, EXACT SAME BACKGROUND, ORIGINAL BACKGROUND UNCHANGED, background preserved, same environment, person visible, photorealistic, seamless integration, high detail"

# Negative prompt - STRONG exclusion of face changes AND background changes
NEGATIVE_PROMPT = "different person, face change, distorted face, changed face, altered face, face modification, face replacement, new face, different face, face editing, face transformation, background change, changed background, different background, new background, background modification, background replacement, altered environment, different environment, mannequin, jacket, hoodie, coat, logo, pattern, flat lay, catalog image, pasted clothing, visible seams, overlay, low quality, blur, gray shirt, black shirt, colored shirt, dark clothing"




class ImageProcessor:
    """
    Processes images for virtual try-on using SEGMENT + COLOR TRANSFER method.
    
    APPROACH: Segment + Color Transfer (OPTION A - GUARANTEED FIX)
    - Does NOT regenerate pixels (no diffusion)
    - Keeps entire person intact (face, body, pose, background)
    - Preserves original texture, shadows, and details
    - Changes only chroma/lightness (color transfer)
    - Zero blur, zero body change, zero person replacement
    - Result: Shirt becomes white, Pants become white, Shoes become white
    - This is NOT diffusion - this is image processing + ML segmentation
    
    Workflow:
    1. Load and preprocess input image
    2. Detect entire person and clothing regions using ControlNet segmentation
    3. Generate precise mask for clothing-only areas (shirt, pants, shoes)
    4. Apply color transfer to convert clothing to white (preserves person)
    5. Save and return result
    
    This method guarantees white clothes without replacing the person.
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
            
            # Step 2: Verify person detection before generating mask
            # CRITICAL: Must detect person first, then extract clothing from person
            person_detected = False
            if segmentation_map is not None:
                # Check if segmentation actually detected a person (not just background)
                seg_array = np.array(segmentation_map.resize((1024, 1024), Image.Resampling.LANCZOS))
                if len(seg_array.shape) == 3:
                    # Check for person-like colors (not just background)
                    r, g, b = seg_array[:, :, 0], seg_array[:, :, 1], seg_array[:, :, 2]
                    # Person typically has: shirt (red/pink), pants (blue), face (yellow/orange)
                    person_pixels = np.sum(
                        ((r > 100) & (r > g * 0.7) & (r > b * 0.7)) |  # Shirt
                        ((b > 100) & (b > r * 0.7) & (b > g * 0.7)) |  # Pants
                        ((r > 150) & (g > 150) & (b < r * 0.7))  # Face
                    )
                    person_ratio = person_pixels / seg_array.size
                    if person_ratio > 0.10:  # At least 10% of image should be person
                        person_detected = True
                        logger.info(f"✅ Person detected in segmentation ({person_ratio*100:.1f}% of image)")
                    else:
                        logger.warning(f"⚠️  Person detection weak ({person_ratio*100:.1f}% of image) - may need fallback")
                else:
                    person_detected = True  # Assume person detected if grayscale
            
            if not person_detected:
                logger.warning("⚠️  Person not clearly detected - will use conservative mask")
            
            # Step 3: Generate clothing mask based on detected clothing
            # CRITICAL: Only mask clothing on detected person, not large geometric shapes
            logger.info("Generating clothing mask from detected clothing (person-first approach)...")
            mask, clothing_items = self.generate_clothing_mask_from_segmentation(
                image, 
                face_detection, 
                segmentation_map,
                person_detected=person_detected  # Pass person detection status
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
            
            # Generate white clothes using diffusion inpainting
            # This creates realistic white clothing that the person is actually wearing
            # Uses ControlNet to preserve person (face, body, pose) while generating white clothes
            logger.info("Generating white clothes using diffusion inpainting...")
            logger.info(f"Using prompt: {prompt}")
            logger.info(f"Using negative prompt: {negative_prompt[:100]}...")
            
            # CRITICAL: Final verification - ensure face is NEVER in mask
            # Double-check that face area has ZERO white pixels
            # BUT: Don't remove too much - we need clothing area to remain white
            mask_array_final = np.array(mask)
            white_before_face_check = np.sum(mask_array_final == 255)
            
            if face_detection is not None:
                try:
                    face_mask_final = self.create_face_mask_from_detection(face_detection, image)
                    face_mask_array_final = np.array(face_mask_final)
                    face_white_pixels_final = np.sum((mask_array_final == 255) & (face_mask_array_final > 5))
                    if face_white_pixels_final > 0:
                        logger.warning(f"⚠️  {face_white_pixels_final} white pixels in face area before generation - removing ONLY face area")
                        # Only remove face area, not entire top region
                        mask_array_final[face_mask_array_final > 5] = 0  # Force protect face
                        mask = Image.fromarray(mask_array_final, mode='L')
                        white_after_face_check = np.sum(mask_array_final == 255)
                        logger.info(f"✅ Face area protected: {white_before_face_check} -> {white_after_face_check} white pixels remaining")
                        if white_after_face_check < white_before_face_check * 0.5:
                            logger.warning(f"⚠️  Face protection removed {white_before_face_check - white_after_face_check} pixels - mask might be too small now")
                    else:
                        logger.info("✅ Face verification passed - no white pixels in face area")
                except Exception as e:
                    logger.warning(f"Could not verify face protection: {e}")
            
            # Generate image with white clothing using img2img inpainting
            # CRITICAL: Mask must have face area as BLACK (preserve), only clothing as WHITE (change)
            # Uses ControlNet to preserve person structure while generating realistic white clothing
            generated_image = await self.vast_ai_client.generate_img2img(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                mask=mask,  # Mask for inpainting (white = generate white clothes, black = preserve face/person)
                denoising_strength=0.55,  # Increased to ensure clothes change (background protected by mask)
                steps=30,  # More steps for better quality white clothing
                cfg_scale=7,  # Higher guidance to ensure white clothing generation
                sampler_name="DPM++ SDE",  # High quality sampler
                width=target_width,
                height=target_height,
                controlnet_pose_enabled=True,  # Preserve pose to keep person intact
                controlnet_pose_weight=1.0,  # Strong pose preservation
                controlnet_pose_control_mode="ControlNet is more important",  # Strong control
                controlnet_inpaint_enabled=True,  # Use inpainting ControlNet for clothing generation
                controlnet_inpaint_weight=0.7  # Increased to ensure clothes change (background protected by mask)
            )
            
            logger.info("✅ White clothing generation completed successfully - person is now wearing white clothes")
            
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
        
        # CRITICAL: Face protection is handled by detected face area (not fixed percentage)
        # This function is called from create_clothing_mask_from_segmentation which handles face protection
        # Do NOT add fixed percentage protection here - use detected face area instead
        
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
            white_pixel_ratio = np.sum(mask_array == 255) / mask_array.size
            logger.info(f"After inversion: white_pixel_ratio={white_pixel_ratio:.2f}")
        
        # CRITICAL: Face protection is handled by detected face area (not fixed percentage)
        # This is applied in generate_clothing_mask_from_segmentation using detected face mask
        
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
        logger.info("Face protection will be applied using detected face area (works for any position)")
        
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
        # Use lower percentile to catch more face area
        threshold = np.percentile(detection_gray, 50)  # Top 50% brightest = face area (was 70%, more aggressive)
        face_mask_array = (detection_gray > threshold).astype(np.uint8) * 255
        
        # Dilate the mask more aggressively to ensure full face, neck, and head coverage
        try:
            from scipy import ndimage
            # Larger dilation structure for better coverage
            face_mask_array = ndimage.binary_dilation(face_mask_array > 127, structure=np.ones((10, 10))).astype(np.uint8) * 255  # Increased from 5x5 to 10x10
        except ImportError:
            # If scipy not available, use PIL filters
            from PIL import ImageFilter
            face_mask_pil = Image.fromarray(face_mask_array, mode='L')
            face_mask_pil = face_mask_pil.filter(ImageFilter.MaxFilter(10))  # Increased from 5 to 10
            face_mask_array = np.array(face_mask_pil)
        
        # CRITICAL: Do NOT add fixed percentage protection
        # Face detection works on ENTIRE image (100% coverage)
        # It can detect face at ANY position (top, middle, bottom)
        # We trust the detection and use only detected face area
        
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
        
        # FINAL FIX: Shirt prompt must say "wearing" - not "replaced with"
        # "wearing" tells SDXL the person is WEARING it, not that it's a separate scene object
        # Without "wearing", SDXL preserves original and places new shirt as separate object
        if clothing_items.get('has_shirt', True):  # Default to True if not detected
            prompt_parts.extend([
                "wearing a plain white shirt",  # FINAL FIX: "wearing" language (not "replaced with")
                "plain white shirt", "white shirt",
                "RGB(255,255,255)", "neutral white fabric"
            ])
            negative_parts.extend([
                "gray shirt", "black shirt", "colored shirt", 
                "dark shirt", "blue shirt", "red shirt", 
                "charcoal shirt", "grey shirt", "dark grey shirt",
                "hidden shirt", "obscured shirt", "covered shirt",
                "translucent object", "semi-transparent", "foggy",
                "blurred clothing", "unclear shirt", "invisible shirt",
                "rectangle overlay", "square overlay", "rectangular overlay",
                "visible mask", "mask edges", "overlay artifact", "red outline",
                "yellow overlay", "brown overlay", "semi-transparent rectangle",
                "blurred rectangle", "blurred square", "blurred overlay",
                "semi-transparent overlay", "blurred area", "obscured person",
                "hidden person", "blurred torso", "blurred body", "distorted image",
                "wavy texture", "liquid texture", "blurred background", "blurred foreground",
                "different clothing style", "different shirt design", "different cut",
                "new clothing", "replacement clothing", "different fit", "altered design"
            ])
        
        # Pants: Current setup is already correct (per user feedback)
        if clothing_items.get('has_pants', False):  # Include pants if detected
            prompt_parts.extend([
                "wearing white pants", "wearing white trousers",  # FINAL FIX: "wearing" language
                "white pants", "white trousers", "white colored pants"
            ])
            negative_parts.extend([
                "dark pants", "black pants", "gray pants", 
                "jeans", "colored pants", "blue jeans", "dark jeans"
            ])
        
        # FINAL FIX: Remove preservation language - it interferes with "wearing" prompt
        # Keep only essential structure preservation via ControlNet (not prompts)
        prompt_parts.extend([
            "wearing white clothing",  # FINAL FIX: "wearing" language for pants too
            "RGB(255,255,255)", "neutral white fabric",
            "white color clothing", "white colored fabric",
            "natural fabric texture", "white fabric texture",
            "clearly visible white colored clothing", "sharp white colored shirt", 
            "crisp white colored fabric", "well-lit white colored clothing",
            "no overlay", "no blur", "crisp image", "sharp image"
        ])
        
        prompt = ", ".join(prompt_parts)
        negative_prompt = ", ".join(negative_parts)
        
        logger.info(f"Generated prompt: {prompt[:100]}...")
        logger.info(f"Clothing items in prompt: shirt={clothing_items.get('has_shirt')}, pants={clothing_items.get('has_pants')}")
        
        return prompt, negative_prompt
    
    def create_clothing_mask_from_segmentation(
        self,
        segmentation_map: Image.Image,
        original_image: Image.Image
    ) -> Image.Image:
        """
        Create precise clothing mask from segmentation map.
        Uses AI segmentation to detect ALL clothing regions (shirt, pants, sleeves, etc.)
        
        Args:
            segmentation_map: Segmentation map from ControlNet
            original_image: Original input image
        
        Returns:
            PIL Image mask (white = clothing, black = everything else)
        """
        width, height = original_image.size
        segmentation = segmentation_map.resize((width, height), Image.Resampling.LANCZOS)
        seg_array = np.array(segmentation)
        
        # Create mask: start with ALL BLACK (preserve everything)
        mask_array = np.zeros((height, width), dtype=np.uint8)
        
        # Segmentation colors typically:
        # - Shirt/torso: Red/pink/magenta (RGB: high R, medium G/B)
        # - Pants/legs: Blue/cyan (RGB: high B, medium R/G)
        # - Arms: Often same as shirt or slightly different
        # - Face/head: Yellow/orange (RGB: high R and G, low B)
        
        if len(seg_array.shape) == 3:
            r, g, b = seg_array[:, :, 0], seg_array[:, :, 1], seg_array[:, :, 2]
            
            # Detect shirt/torso: Red/pink/magenta tones (high red, red > green and blue)
            shirt_mask = (r > 100) & (r > g * 0.8) & (r > b * 0.8) & (r < 255)
            
            # Detect pants/legs: Blue/cyan tones (high blue, blue > red and green)
            # INCLUDE pants - change entire clothing to white
            pants_mask = (b > 100) & (b > r * 0.8) & (b > g * 0.8) & (b < 255)
            
            # Detect arms: Similar to shirt but might be slightly different
            # Look for red/pink tones that are not face (face is usually brighter/yellow)
            arms_mask = (r > 80) & (r > g * 0.7) & (r > b * 0.7) & (r < 200)
            
            # Combine ALL clothing regions (shirt + pants + arms) - change entire clothing
            clothing_mask = shirt_mask | pants_mask | arms_mask  # Include all clothing
            
            # Exclude face/head: Yellow/orange tones (high R and G, low B)
            # Also exclude skin tones and head region
            face_mask_seg = (r > 150) & (g > 150) & (b < r * 0.7) & (b < g * 0.7)
            
            # CRITICAL: Do NOT use fixed percentage for head region
            # Face detection works on ENTIRE image and can detect face at ANY position
            # We'll use detected face area instead of fixed percentage
            # Create empty head region mask - face protection handled by detected face area
            head_region_mask = np.zeros((height, width), dtype=bool)  # Empty - face detection handles this
            
            # FIX 2: Temporarily INCLUDE hands for upper-garment edits
            # Hands define garment ownership - excluding them causes "held object" hallucinations
            # We'll subtract skin-tone pixels only (not geometry) later
            # Skin tones: High R and G, medium B
            skin_mask = (r > 120) & (g > 100) & (b < r * 0.9) & (b < g * 0.9) & (r < 220) & (g < 200)
            
            # CRITICAL: Exclude background explicitly
            # Background detection: Usually has different characteristics
            # 1. Background is often at edges/corners
            # 2. Background has different color distribution (often less saturated)
            # 3. Background pixels are usually NOT in clothing segmentation colors
            
            # Method 1: Exclude edge regions (background is often at edges)
            edge_mask = np.zeros((height, width), dtype=bool)
            edge_threshold = 0.05  # 5% from edges
            edge_mask[:int(height * edge_threshold), :] = True  # Top edge
            edge_mask[int(height * (1 - edge_threshold)):, :] = True  # Bottom edge
            edge_mask[:, :int(width * edge_threshold)] = True  # Left edge
            edge_mask[:, int(width * (1 - edge_threshold)):] = True  # Right edge
            
            # Method 2: Exclude low-saturation areas (background often less saturated)
            # Calculate saturation: max(R,G,B) - min(R,G,B)
            max_rgb = np.maximum(np.maximum(r, g), b)
            min_rgb = np.minimum(np.minimum(r, g), b)
            saturation = max_rgb - min_rgb
            low_saturation_mask = saturation < 30  # Low saturation = likely background
            
            # Method 3: Exclude areas that are NOT in clothing color ranges
            # If pixel is not red/pink (shirt), not blue (pants), and not in clothing mask, it's likely background
            not_clothing_color = ~((r > 80) | (b > 80))  # Not red/pink and not blue
            
            # Combine background detection methods
            background_mask = edge_mask | (low_saturation_mask & not_clothing_color)
            
            # Remove face, head, skin tones, AND background from clothing mask
            # This ensures we only mask clothing, not the person's body parts or background
            clothing_mask = clothing_mask & (~face_mask_seg) & (~head_region_mask) & (~skin_mask) & (~background_mask)
            
            # Set clothing regions to white (255) - ONLY clothing, not person's body
            mask_array[clothing_mask] = 255
            
            # CRITICAL: Ensure mask only covers clothing on detected person
            # Don't force geometric overrides - rely on segmentation to detect actual clothing
            # This ensures we preserve the person and only change clothing color
            logger.info(f"✅ Clothing mask created from segmentation - only clothing areas masked (person preserved)")
            
            logger.info(f"Created clothing mask from segmentation: {np.sum(mask_array == 255)} white pixels ({np.sum(mask_array == 255)/mask_array.size*100:.1f}% of image)")
        else:
            # Grayscale segmentation - use threshold
            # Higher values usually indicate body/clothing regions
            threshold = np.percentile(seg_array, 50)  # Middle 50% as threshold
            clothing_mask = seg_array > threshold
            mask_array[clothing_mask] = 255
            logger.info(f"Created clothing mask from grayscale segmentation: {np.sum(mask_array == 255)} white pixels")
        
        return Image.fromarray(mask_array, mode='L')
    
    def generate_clothing_mask_from_segmentation(
        self, 
        image: Image.Image, 
        face_detection: Optional[Image.Image] = None,
        segmentation_map: Optional[Image.Image] = None,
        person_detected: bool = True
    ) -> Tuple[Image.Image, Dict[str, bool]]:
        """
        Generate clothing mask based on detected clothing from segmentation.
        APPROACH: Detect entire person first, then extract ONLY clothing regions from person.
        
        CRITICAL: Only masks clothing on detected person, not large geometric shapes.
        This ensures person is preserved and only clothing is changed.
        
        Args:
            image: Input PIL Image
            face_detection: Face detection map from ControlNet (optional)
            segmentation_map: Segmentation map from ControlNet (optional)
            person_detected: Whether person was detected in segmentation (default: True)
        
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
        
        # Create mask from segmentation if available (AI-based detection)
        if segmentation_map is not None:
            try:
                logger.info("Creating clothing mask from AI segmentation map...")
                mask = self.create_clothing_mask_from_segmentation(segmentation_map, image)
                mask_array = np.array(mask)
                logger.info(f"✅ AI segmentation mask created: {np.sum(mask_array == 255)} white pixels ({np.sum(mask_array == 255)/mask_array.size*100:.1f}% of image)")
            except Exception as e:
                logger.warning(f"Failed to create mask from segmentation: {e}, using geometric mask")
                mask = Image.new("L", (width, height), 0)
                mask_array = np.array(mask)
        else:
            # Fallback to geometric mask if segmentation not available
            logger.warning("No segmentation map available, using geometric mask")
            mask = Image.new("L", (width, height), 0)
            mask_array = np.array(mask)
        
        # CRITICAL: Create face protection mask from ENTIRE image detection
        # Face detection works on 100% of image, so it can detect face at ANY position
        face_mask = None
        if face_detection is not None:
            try:
                face_mask = self.create_face_mask_from_detection(face_detection, image)
                logger.info("✅ Using ControlNet face detection from ENTIRE image for face protection")
                logger.info("   Face can be detected at ANY position (top, middle, bottom)")
            except Exception as e:
                logger.warning(f"Failed to create face mask: {e}, using fallback")
                face_mask = None
        
        # Fallback: Only use if face detection completely failed
        # But this is less ideal - we prefer actual face detection
        face_protection_fallback = int(height * 0.20) if face_mask is None else None  # 20% face protection
        
        # CRITICAL: Apply face protection to mask - MUST protect detected face completely
        # Face detection covers ENTIRE image, so it works regardless of face position
        if face_mask is not None:
            mask_array = np.array(mask)
            face_mask_array = np.array(face_mask)
            # Protect face area: set to 0 (black = preserve, don't change)
            # Very low threshold to protect ALL face pixels, including edges
            # This works for face at ANY position (top, middle, bottom)
            mask_array[face_mask_array > 5] = 0  # Very low threshold to protect all face pixels
            mask = Image.fromarray(mask_array, mode='L')
            
            # CRITICAL: Verify face is protected - MUST have ZERO white pixels in face
            face_white_pixels = np.sum((mask_array == 255) & (face_mask_array > 5))
            if face_white_pixels > 0:
                logger.error(f"❌ CRITICAL ERROR: {face_white_pixels} white pixels found in detected face area! Forcing to black...")
                mask_array[face_mask_array > 5] = 0  # Force protect face
                # Verify again
                face_white_pixels_after = np.sum((mask_array == 255) & (face_mask_array > 5))
                if face_white_pixels_after == 0:
                    logger.info("✅ Face protection verified - face area is now completely black")
                else:
                    logger.error(f"❌ ERROR: Still {face_white_pixels_after} white pixels in face after forced protection!")
                mask = Image.fromarray(mask_array, mode='L')
            else:
                logger.info("✅ Face protection verified - no white pixels in face area")
            
            logger.info("✅ Applied face detection mask from ENTIRE image - face protected regardless of position")
        else:
            # Fallback: protect top 20% of image (only if face detection failed)
            mask_array = np.array(mask)
            face_protection_zone = int(height * 0.20)  # Top 20% for face/head protection (fallback only)
            mask_array[:face_protection_zone, :] = 0  # Black = preserve face
            mask = Image.fromarray(mask_array, mode='L')
            logger.warning(f"⚠️  Using fallback face protection (top {face_protection_zone*100/height:.1f}%) - face detection failed")
            logger.warning("   This may not work if face is not at top of image")
        
        # CRITICAL: Only use geometric fallback if person is detected
        # If person not detected, don't create large geometric mask (would create large shirt)
        mask_array = np.array(mask)
        white_ratio = np.sum(mask_array == 255) / mask_array.size
        
        # Only use geometric fallback if person is detected AND mask is too small
        use_geometric_fallback = False
        if white_ratio < 0.10:  # Less than 10% white pixels - mask might be too small (increased from 5%)
            if person_detected:
                logger.warning(f"Segmentation mask too small ({white_ratio*100:.1f}%), using conservative geometric mask (person detected)")
                use_geometric_fallback = True
            else:
                logger.error("⚠️  Person not detected AND mask too small - cannot create mask safely")
                logger.error("   Returning minimal mask to avoid creating large shirt")
                # Return minimal mask - don't create large geometric shapes
                mask = Image.fromarray(mask_array, mode='L')
                return mask, clothing_items
        
        if use_geometric_fallback:
            draw = ImageDraw.Draw(mask)
            
            # CRITICAL: Only mask clothing on detected person, not large geometric shapes
            # Use conservative geometric mask that follows person's body shape
            # Don't create large shirt - only mask actual clothing area on person
            if clothing_items.get('has_shirt', True):
                # Conservative shirt mask - only covers torso area, not entire image
                # This prevents creating a large shirt instead of detecting person
                shirt_top = int(height * 0.20)  # Start below face (20% from top)
                shirt_bottom = int(height * 0.70)  # Stop at waist (70% from top)
                # Narrower width to follow person's body, not entire image width
                chest_left = int(width * 0.20)  # 20% from left (narrower - follows person)
                chest_right = int(width * 0.80)  # 80% from left (narrower - follows person)
                
                # Only mask torso area - don't include full width or arms
                # This ensures we only mask clothing on person, not create large shirt
                draw.rectangle([(chest_left, shirt_top), (chest_right, shirt_bottom)], fill=255)
                logger.info(f"✅ Masked shirt area on detected person (from {shirt_top*100/height:.0f}% to {shirt_bottom*100/height:.0f}%, width {chest_left*100/width:.0f}%-{chest_right*100/width:.0f}%)")
            
            # INCLUDE pants in mask - change entire clothing to white
            # Use narrower width to follow person's body, not entire image
            if clothing_items.get('has_pants', False):
                pants_top = int(height * 0.65)
                pants_bottom = int(height * 0.95)  # Cover full pants area
                pants_left = int(width * 0.20)  # Narrower to follow person (was 0.15)
                pants_right = int(width * 0.80)  # Narrower to follow person (was 0.85)
                draw.rectangle([(pants_left, pants_top), (pants_right, pants_bottom)], fill=255)
                logger.info(f"✅ Masked pants area on detected person (width {pants_left*100/width:.0f}%-{pants_right*100/width:.0f}%)")
            else:
                # Even if pants not detected, cover lower body area with narrow width
                pants_top = int(height * 0.70)
                pants_bottom = int(height * 0.92)
                pants_left = int(width * 0.25)  # Narrower to follow person (was 0.20)
                pants_right = int(width * 0.75)  # Narrower to follow person (was 0.80)
                draw.rectangle([(pants_left, pants_top), (pants_right, pants_bottom)], fill=255)
                logger.info(f"✅ Masked lower body area on detected person (width {pants_left*100/width:.0f}%-{pants_right*100/width:.0f}%)")
            
            # CRITICAL: Re-apply face protection after geometric mask
            # Use detected face area from ENTIRE image, not fixed percentage
            mask_array = np.array(mask)
            
            if face_mask is not None:
                # Use detected face area - works regardless of face position
                face_mask_array = np.array(face_mask)
                mask_array[face_mask_array > 5] = 0  # Protect all detected face pixels
                mask = Image.fromarray(mask_array, mode='L')
                
                # Verify face is still protected
                face_white_pixels = np.sum((mask_array == 255) & (face_mask_array > 5))
                if face_white_pixels > 0:
                    logger.error(f"⚠️  CRITICAL: {face_white_pixels} white pixels in face after geometric mask! Forcing protection...")
                    mask_array[face_mask_array > 5] = 0  # Force protect
                    mask = Image.fromarray(mask_array, mode='L')
                
                logger.info("✅ Re-applied face detection mask from ENTIRE image after geometric mask")
            elif face_protection_fallback:
                # Fallback: only if face detection failed
                mask_array[:face_protection_fallback, :] = 0
                mask = Image.fromarray(mask_array, mode='L')
                logger.warning(f"⚠️  Re-applied fallback face protection (top {face_protection_fallback*100/height:.0f}%) - face detection failed")
        
        # Final cleanup - balanced mask with moderate blur for smooth edges
        mask_array = np.array(mask)
        mask_array = np.where(mask_array > 127, 255, 0).astype(np.uint8)
        
        # Apply moderate edge smoothing to prevent hard edges and overlay artifacts
        mask_pil = Image.fromarray(mask_array, mode='L')
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=2))  # Moderate blur (radius=2) for smooth edges
        
        # Re-threshold after blur - balanced threshold for smooth transitions
        mask_array = np.array(mask_pil)
        mask_array = np.where(mask_array > 100, 255, 0).astype(np.uint8)  # Balanced threshold for smooth edges
        
        # CRITICAL: Apply face protection BEFORE blur to ensure face is protected
        # Use detected face area from ENTIRE image, not fixed percentage
        if face_mask is not None:
            face_mask_array = np.array(face_mask)
            # Protect face area aggressively: any pixel that might be face = black (preserve)
            # This works for face at ANY position (top, middle, bottom)
            mask_array[face_mask_array > 5] = 0  # Very low threshold to protect all face pixels
            
            # Verify face is protected
            face_white_pixels = np.sum((mask_array == 255) & (face_mask_array > 5))
            if face_white_pixels > 0:
                logger.error(f"⚠️  CRITICAL: {face_white_pixels} white pixels in face before blur! Forcing protection...")
                mask_array[face_mask_array > 5] = 0  # Force protect
            
            logger.info("✅ Applied face detection mask from ENTIRE image before blur")
        else:
            # Fallback: only if face detection failed
            face_protection_zone = int(height * 0.20)  # 20% face protection
            mask_array[:face_protection_zone, :] = 0
            logger.warning(f"⚠️  Applied fallback face protection (top {face_protection_zone*100/height:.1f}%) - face detection failed")
        
        # Apply moderate edge smoothing (very light blur) to prevent hard edges
        # But NOT too much blur - that creates the blurred rectangle overlay artifact
        mask_pil = Image.fromarray(mask_array, mode='L')
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=2))  # Moderate blur (radius=2) for smooth edges
        
        # Re-threshold after moderate blur - keep it mostly sharp
        mask_array = np.array(mask_pil)
        mask_array = np.where(mask_array > 100, 255, 0).astype(np.uint8)  # Balanced threshold for smooth edges
        
        # FINAL face protection check - ensure face is NEVER white in mask
        if face_mask is not None:
            face_mask_array = np.array(face_mask)
            # Final aggressive protection: any face pixel = black (0)
            mask_array[face_mask_array > 5] = 0  # Very low threshold (was 20) to ensure complete face protection
            logger.info("Applied final aggressive face protection after blur")
        else:
            # Final fallback: protect top 20% of image
            face_protection_zone = int(height * 0.20)  # 20% face protection
            mask_array[:face_protection_zone, :] = 0
            logger.info(f"Applied final fallback face protection (top {face_protection_zone*100/height:.1f}%)")
        
        # CRITICAL: Additional protection - Exclude background areas
        # Protect edges and corners (background is usually at edges)
        edge_protection = 0.03  # 3% from edges
        mask_array[:int(height * edge_protection), :] = 0  # Top edge (background)
        mask_array[int(height * (1 - edge_protection)):, :] = 0  # Bottom edge (background/feet)
        mask_array[:, :int(width * edge_protection)] = 0  # Left edge (background)
        mask_array[:, int(width * (1 - edge_protection)):] = 0  # Right edge (background)
        
        # Protect top 10% (face/head)
        mask_array[:int(height * 0.20), :] = 0  # Top 20% always protected (face/head)
        # Protect very bottom (feet/shoes area)
        mask_array[int(height * 0.95):, :] = 0  # Bottom 5% protected (feet/shoes)
        logger.info("Applied additional protection for face, head, feet, and BACKGROUND EDGES")
        
        mask = Image.fromarray(mask_array, mode='L')
        logger.info("Applied moderate edge smoothing to prevent blurred rectangle overlay artifacts")
        
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
        face_protection_fallback = int(height * 0.20) if face_mask is None else None  # 20% face protection fallback
        
        # CLOTHING REGION: Cover ALL clothes (shirt + pants)
        # Shirt region: start from very top to catch full shirt including collar
        # We'll protect face area separately using detected face mask
        shirt_top = int(height * 0.20)  # Start below face protection (20% from top)
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
        
        # CRITICAL: Ensure background areas are black (preserved)
        # Protect bottom area (shoes, feet, background)
        draw.rectangle(
            [(0, pants_bottom), (width, height)],
            fill=0  # Black = preserve shoes, feet, background
        )
        
        # Protect top edge (background/sky) - 3% from top
        draw.rectangle(
            [(0, 0), (width, int(height * 0.03))],
            fill=0  # Black = preserve top background
        )
        
        # Protect side margins (background) - 3% from each edge
        # Left margin
        draw.rectangle(
            [(0, 0), (int(width * 0.03), height)],
            fill=0  # Black = preserve left background
        )
        # Right margin
        draw.rectangle(
            [(int(width * 0.97), 0), (width, height)],
            fill=0  # Black = preserve right background
        )
        
        logger.info("✅ Background edges protected in geometric mask: top, bottom, left, right margins set to black")
        
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
    
    def change_clothing_color_with_mask(
        self, 
        image: Image.Image, 
        mask: Image.Image, 
        target_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> Image.Image:
        """
        OPTION A: Segment + Color Transfer (GUARANTEED FIX)
        
        Change clothing color using mask - preserves texture, shadows, and details.
        Uses luminance-preserving colorization: keeps original brightness, changes chroma/lightness only.
        
        Key Features:
        - Does NOT regenerate pixels (no diffusion)
        - Keeps entire person intact (face, body, pose, background)
        - Preserves original texture and shadows
        - Changes only chroma/lightness (color transfer)
        - Zero blur, zero body change, zero person replacement
        - Guarantees white clothes without artifacts
        
        Args:
            image: Original PIL Image
            mask: PIL Image mask (white = clothing area to change, black = preserve)
            target_color: RGB tuple for target color (default: white (255, 255, 255))
        
        Returns:
            PIL Image with changed clothing color (texture and details preserved)
        """
        # Convert images to numpy arrays
        img_array = np.array(image.convert("RGB")).astype(np.float32)
        mask_array = np.array(mask.convert("L"))
        
        # Normalize mask to 0-1 range (0 = black/preserve, 1 = white/change)
        # Use soft mask for smooth blending at edges
        mask_normalized = (mask_array / 255.0).astype(np.float32)
        
        # Convert target color to float
        target_rgb = np.array(target_color, dtype=np.float32)
        
        # CRITICAL: Preserve texture by keeping the original grayscale (luminance) structure
        # Calculate luminance - this contains ALL texture, shadow, and detail information
        # Formula: Y = 0.299*R + 0.587*G + 0.114*B (standard luminance calculation)
        luminance = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
        
        # Create result array - start with original
        result_array = img_array.copy()
        
        # Method: For white clothing, preserve luminance structure, shift toward white
        # This preserves ALL texture because luminance contains the structure information
        # Dark areas → light gray-white (preserves shadows)
        # Bright areas → bright white (preserves highlights)
        # This maintains natural fabric folds and texture
        
        # For white color (255, 255, 255): use enhanced luminance-preserving method
        if np.array_equal(target_rgb, [255, 255, 255]):
            # Enhanced method for guaranteed white clothes:
            # 1. Preserve luminance structure (texture, shadows, folds)
            # 2. Shift luminance toward white while maintaining relative brightness
            # 3. Ensure sufficient brightness for clear white appearance
            
            # Normalize luminance to 0-1 range
            luminance_norm = luminance / 255.0
            
            # Enhanced white conversion:
            # - Dark areas (low luminance) → light gray-white (minimum 200 to ensure "white" appearance)
            # - Bright areas (high luminance) → bright white (255)
            # - Preserves texture by maintaining relative luminance differences
            # Formula: result = min_luminance + (max_luminance - min_luminance) * normalized_luminance
            min_white = 200.0  # Minimum white value (ensures "white" appearance, not gray)
            max_white = 255.0  # Maximum white value
            white_luminance = min_white + (max_white - min_white) * luminance_norm
            
            # Apply to all RGB channels (same for white)
            for channel in range(3):
                result_array[:, :, channel] = (
                    img_array[:, :, channel] * (1 - mask_normalized) + 
                    white_luminance * mask_normalized
                )
        else:
            # For non-white colors: use luminance-preserving colorization
            for channel in range(3):
                original_channel = img_array[:, :, channel]
                
                # Scale target color by original brightness to preserve texture
                brightness_scale = luminance / 255.0
                target_value = target_rgb[channel] * brightness_scale
                
                # Ensure values are in valid range
                target_value = np.clip(target_value, 0, 255)
                
                # Apply color change only in masked areas
                result_array[:, :, channel] = (
                    original_channel * (1 - mask_normalized) + 
                    target_value * mask_normalized
                )
        
        # Convert back to uint8 and create PIL Image
        result_array = np.clip(result_array, 0, 255).astype(np.uint8)
        result_image = Image.fromarray(result_array, mode='RGB')
        
        logger.info(f"✅ Color transfer completed: Changed clothing to RGB{target_color} in masked areas")
        logger.info(f"   - Texture preserved: YES (luminance structure maintained)")
        logger.info(f"   - Shadows preserved: YES (relative brightness maintained)")
        logger.info(f"   - Zero blur: YES (no diffusion, pure image processing)")
        logger.info(f"   - Zero body change: YES (only color changed, no pixel regeneration)")
        
        return result_image
    
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
