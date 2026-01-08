"""
Image Processor for Virtual Try-On
Handles image processing, masking, and SDXL-based clothing generation.
"""

import io
import base64
import numpy as np
from PIL import Image
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Processes images for virtual try-on using SDXL inpainting.
    
    Workflow:
    1. Load and preprocess input image
    2. Generate mask for clothing area (torso/body)
    3. Create prompt based on selected style
    4. Use SDXL inpainting to generate new clothing
    5. Post-process and return result
    """
    
    def __init__(self, model_manager):
        """
        Initialize ImageProcessor with model manager.
        
        Args:
            model_manager: ModelManager instance for accessing SDXL models
        """
        self.model_manager = model_manager
        self.temp_dir = Path("temp_images")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Style-specific prompts for better results
        self.style_prompts = {
            "casual": "a person wearing casual, comfortable everyday clothing, t-shirt or casual shirt, jeans or casual pants, relaxed fit, natural lighting, realistic",
            "formal": "a person wearing formal business attire, professional suit or dress shirt, elegant and sophisticated, proper fit, professional lighting, realistic",
            "sportswear": "a person wearing athletic sportswear, activewear, sports t-shirt or tank top, athletic shorts or leggings, fitness clothing, dynamic lighting, realistic",
            "streetwear": "a person wearing trendy streetwear, urban fashion, hoodie or street style top, streetwear pants or joggers, modern urban style, street lighting, realistic",
            "random": "a person wearing stylish, well-fitted clothing, contemporary fashion, good quality fabric, proper lighting and shadows, realistic, high quality"
        }
    
    async def process_image(self, image_data: bytes, style: str) -> Dict:
        """
        Main processing function: takes image and style, returns generated image.
        
        Args:
            image_data: Raw image bytes from upload
            style: Selected clothing style
        
        Returns:
            Dictionary with success status and image URL or error message
        """
        try:
            # Load image from bytes
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            
            # Resize image if too large (SDXL works best with 1024x1024)
            # Maintain aspect ratio
            max_size = 1024
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Generate mask for clothing area (torso/body region)
            mask = self.generate_clothing_mask(image)
            
            # Get style-specific prompt
            prompt = self.style_prompts.get(style, self.style_prompts["random"])
            
            # Add instruction to keep face and hair unchanged
            full_prompt = f"{prompt}, keeping the person's face and hair completely unchanged, maintaining original facial features and hairstyle"
            
            # Negative prompt (what we don't want)
            negative_prompt = "blurry, distorted face, changed facial features, different hair, deformed, ugly, bad quality, artifacts"
            
            # Generate image using SDXL inpainting
            generated_image = await self.generate_with_sdxl(
                image, 
                mask, 
                full_prompt, 
                negative_prompt,
                denoise_strength=0.65  # Adjustable: 0.50, 0.65, 0.75
            )
            
            # Save generated image temporarily
            filename = f"generated_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}.png"
            filepath = self.temp_dir / filename
            generated_image.save(filepath, "PNG")
            
            # Clean up old files (keep only last 10)
            self.cleanup_old_files(keep_count=10)
            
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
        Generate a mask for the clothing area (torso/body region).
        
        This is a simplified mask generation. In production, you might want to use:
        - Human pose estimation (MediaPipe, OpenPose)
        - Semantic segmentation (DeepLab, Segment Anything)
        - Manual annotation tools
        
        For now, we create a simple rectangular mask for the torso area.
        
        Args:
            image: Input PIL Image
        
        Returns:
            PIL Image mask (white = area to inpaint, black = keep original)
        """
        width, height = image.size
        
        # Create mask: focus on torso area (middle section of image)
        # Adjust these ratios based on your needs
        mask = Image.new("L", (width, height), 0)  # Start with all black (keep original)
        
        # Define torso region (middle 40-60% of image height, full width)
        # This is a simple approximation - in production, use pose/segmentation models
        torso_top = int(height * 0.25)  # Start at 25% from top
        torso_bottom = int(height * 0.75)  # End at 75% from top
        torso_left = int(width * 0.1)  # 10% margin on left
        torso_right = int(width * 0.9)  # 10% margin on right
        
        # Create white region (area to inpaint) with soft edges
        from PIL import ImageDraw, ImageFilter
        
        draw = ImageDraw.Draw(mask)
        
        # Draw rounded rectangle for smoother blending
        draw.rectangle(
            [(torso_left, torso_top), (torso_right, torso_bottom)],
            fill=255  # White = inpaint this area
        )
        
        # Apply Gaussian blur for soft edges (better blending)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=20))
        
        return mask
    
    async def generate_with_sdxl(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: str,
        denoise_strength: float = 0.65,
        num_inference_steps: int = 30
    ) -> Image.Image:
        """
        Generate image using SDXL inpainting pipeline.
        
        Args:
            image: Original image
            mask: Mask indicating area to inpaint (white = inpaint, black = keep)
            prompt: Text prompt describing desired clothing
            negative_prompt: What to avoid in generation
            denoise_strength: How much to change (0.0-1.0), higher = more change
            num_inference_steps: Number of diffusion steps (more = better quality, slower)
        
        Returns:
            Generated PIL Image
        """
        try:
            # Get the inpainting pipeline from model manager
            pipeline = self.model_manager.get_inpainting_pipeline()
            
            # Ensure image and mask are the same size
            if image.size != mask.size:
                mask = mask.resize(image.size, Image.Resampling.LANCZOS)
            
            # Convert to numpy for processing
            # SDXL expects images in specific format
            image_np = np.array(image)
            mask_np = np.array(mask)
            
            # Generate using SDXL inpainting
            # The pipeline handles the actual diffusion process
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask,
                strength=denoise_strength,  # How much to change the masked area
                num_inference_steps=num_inference_steps,
                guidance_scale=7.5,  # How closely to follow the prompt
                generator=torch.Generator(device=self.model_manager.device).manual_seed(42)  # For reproducibility
            )
            
            # Extract generated image from result
            generated_image = result.images[0]
            
            # Post-process: blend with original for smoother result
            # This helps maintain natural look
            final_image = self.blend_images(image, generated_image, mask)
            
            return final_image
            
        except Exception as e:
            logger.error(f"Error in SDXL generation: {e}")
            raise
    
    def blend_images(
        self, 
        original: Image.Image, 
        generated: Image.Image, 
        mask: Image.Image
    ) -> Image.Image:
        """
        Blend generated image with original for smoother transitions.
        
        Args:
            original: Original input image
            generated: Generated image from SDXL
            mask: Mask used for inpainting
        
        Returns:
            Blended PIL Image
        """
        # Convert to numpy arrays for blending
        orig_np = np.array(original, dtype=np.float32)
        gen_np = np.array(generated, dtype=np.float32)
        mask_np = np.array(mask, dtype=np.float32) / 255.0  # Normalize to 0-1
        
        # Expand mask to 3 channels (RGB)
        if len(mask_np.shape) == 2:
            mask_np = np.stack([mask_np] * 3, axis=-1)
        
        # Blend: use generated where mask is high, original where mask is low
        blended = orig_np * (1 - mask_np) + gen_np * mask_np
        
        # Convert back to PIL Image
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        return Image.fromarray(blended)
    
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


