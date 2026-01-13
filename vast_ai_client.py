"""
Vast.ai Stable Diffusion API Client
Handles communication with Vast.ai's Stable Diffusion API for img2img generation.
"""

import base64
import io
import aiohttp
import logging
from typing import Dict, Optional
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VastAIClient:
    """
    Client for Vast.ai Stable Diffusion API.
    
    This client sends img2img requests to Vast.ai's SD API endpoint
    and handles the response.
    """
    
    def __init__(self, api_url: str, timeout: int = 300):
        """
        Initialize Vast.ai API client.
        
        Args:
            api_url: Base URL of Vast.ai SD API (e.g., "http://localhost:8081")
            timeout: Request timeout in seconds (default: 300 for image generation)
        """
        self.api_url = api_url.rstrip('/')
        self.img2img_endpoint = f"{self.api_url}/sdapi/v1/img2img"
        self.controlnet_detect_endpoint = f"{self.api_url}/controlnet/detect"  # For segmentation preprocessing
        self.timeout = timeout
        logger.info(f"Initialized Vast.ai client with API URL: {self.api_url}")
        logger.info(f"ControlNet detect endpoint: {self.controlnet_detect_endpoint}")
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string.
        
        Args:
            image: PIL Image object
        
        Returns:
            Base64 encoded image string
        """
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_base64
    
    def _base64_to_image(self, img_base64: str) -> Image.Image:
        """
        Convert base64 string to PIL Image.
        
        Args:
            img_base64: Base64 encoded image string
        
        Returns:
            PIL Image object
        """
        img_data = base64.b64decode(img_base64)
        return Image.open(io.BytesIO(img_data))
    
    async def get_segmentation_map(self, image: Image.Image) -> Image.Image:
        """
        Get segmentation map from ControlNet segmentation preprocessor.
        This detects and labels body parts in the image.
        
        Args:
            image: Input PIL Image
        
        Returns:
            Segmentation map as PIL Image (colored map with body part labels)
        """
        try:
            image_base64 = self._image_to_base64(image)
            
            payload = {
                "controlnet_module": "segmentation",
                "controlnet_input_images": [image_base64],
                "controlnet_processor_res": 512,
                "controlnet_threshold_a": 64,
                "controlnet_threshold_b": 64
            }
            
            logger.info("Requesting segmentation map from ControlNet...")
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(
                    self.controlnet_detect_endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"ControlNet segmentation error {response.status}: {error_text}")
                    
                    result = await response.json()
                    
                    # Extract segmentation image from response
                    if "images" not in result or len(result["images"]) == 0:
                        raise Exception("No segmentation image returned from ControlNet")
                    
                    segmentation_base64 = result["images"][0]
                    segmentation_image = self._base64_to_image(segmentation_base64)
                    
                    logger.info("Segmentation map received successfully")
                    return segmentation_image
                    
        except Exception as e:
            logger.error(f"Error getting segmentation map: {e}")
            raise
    
    async def test_connection(self) -> bool:
        """
        Test connection to Vast.ai API by checking if endpoint is accessible.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                # Try to access the options endpoint (lightweight check)
                async with session.get(self.health_endpoint) as response:
                    if response.status == 200:
                        logger.info(f"✅ Successfully connected to Vast.ai API at {self.api_url}")
                        return True
                    else:
                        logger.warning(f"⚠️  API returned status {response.status} at {self.health_endpoint}")
                        # Try alternative endpoint
                        alt_endpoint = f"{self.api_url}/sdapi/v1/version"
                        async with session.get(alt_endpoint) as alt_response:
                            if alt_response.status == 200:
                                logger.info(f"✅ Successfully connected via alternative endpoint")
                                return True
                        return False
        except aiohttp.ClientConnectorError as e:
            logger.error(f"❌ Cannot connect to {self.api_url}: {e}")
            logger.error(f"   Please check:")
            logger.error(f"   1. Is SD WebUI running?")
            logger.error(f"   2. Is the URL correct? (current: {self.api_url})")
            logger.error(f"   3. If on same server, use: http://localhost:7860")
            logger.error(f"   4. If remote, use external port: http://74.48.140.178:8081")
            return False
        except Exception as e:
            logger.error(f"❌ Error testing connection: {e}")
            return False
    
    async def generate_img2img(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        mask: Optional[Image.Image] = None,
        denoising_strength: float = 0.65,
        steps: int = 25,
        cfg_scale: float = 5,
        sampler_name: str = "DPM++ SDE",
        width: int = 1024,
        height: int = 1024,
        controlnet_enabled: bool = True,
        controlnet_module: Optional[str] = "none",
        controlnet_model: Optional[str] = "controlnet-inpaint-dreamer-sdxl",
        controlnet_weight: float = 1.0,
        controlnet_control_mode: str = "Balanced",
        controlnet_pixel_perfect: bool = True
    ) -> Image.Image:
        """
        Generate image using Vast.ai img2img API with inpainting mask + ControlNet.
        
        Uses img2img + ControlNet + Mask for precise clothing replacement.
        
        Args:
            image: Input PIL Image
            prompt: Positive prompt for generation
            negative_prompt: Negative prompt (what to avoid)
            mask: Mask image for inpainting (white=change, black=preserve)
            denoising_strength: How much to change the image (0.0-1.0)
            steps: Number of inference steps
            cfg_scale: Guidance scale (how closely to follow prompt)
            sampler_name: Sampler to use (e.g., "DPM++ SDE")
            width: Output image width
            height: Output image height
            controlnet_enabled: Whether to use ControlNet
            controlnet_module: ControlNet module name ("none" for inpaint models)
            controlnet_model: ControlNet model name ("controlnet-inpaint-dreamer-sdxl")
            controlnet_weight: ControlNet weight
            controlnet_control_mode: Control mode ("Balanced" for best results)
            controlnet_pixel_perfect: Enable pixel perfect mode
        
        Returns:
            Generated PIL Image
        """
        try:
            # Convert image to base64
            image_base64 = self._image_to_base64(image)
            
            # Convert mask to base64 if provided
            mask_b64 = None
            if mask is not None:
                mask_b64 = self._image_to_base64(mask)
            else:
                raise Exception("Mask is required for clothing replacement - binary clothes-only mask must be provided")
            
            # Build payload matching exact format
            payload = {
                "init_images": [image_base64],
                "mask": mask_b64,
                "denoising_strength": denoising_strength,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "sampler_name": sampler_name,
                "width": width,
                "height": height,
            }
            
            # Add ControlNet using alwayson_scripts format with exact structure
            if controlnet_enabled and controlnet_model:
                payload["alwayson_scripts"] = {
                    "ControlNet": {
                        "args": [
                            {
                                "enabled": True,
                                "model": controlnet_model,
                                "module": controlnet_module if controlnet_module else "none",
                                "weight": controlnet_weight,
                                "control_mode": controlnet_control_mode,
                                "pixel_perfect": controlnet_pixel_perfect,
                                "resize_mode": "Crop and Resize"
                            }
                        ]
                    }
                }
                logger.info(f"ControlNet enabled: {controlnet_model} / {controlnet_module}")
                logger.info(f"ControlNet weight: {controlnet_weight}, control_mode: {controlnet_control_mode}")
            else:
                logger.warning("ControlNet is disabled")
            
            # Send request to Vast.ai API
            logger.info(f"Sending img2img request to {self.img2img_endpoint}")
            logger.info(f"Prompt: {prompt[:100]}...")
            logger.info(f"Parameters: steps={steps}, denoising_strength={denoising_strength}, cfg_scale={cfg_scale}")
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(
                    self.img2img_endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        error_msg = f"Vast.ai API error {response.status}: {error_text}"
                        logger.error(f"API Request failed:")
                        logger.error(f"  URL: {self.img2img_endpoint}")
                        logger.error(f"  Status: {response.status}")
                        logger.error(f"  Error: {error_text}")
                        raise Exception(error_msg)
                    
                    result = await response.json()
                    
                    # Extract generated image from response
                    if "images" not in result or len(result["images"]) == 0:
                        raise Exception("No images returned from Vast.ai API")
                    
                    # Convert base64 response to PIL Image
                    generated_image_base64 = result["images"][0]
                    generated_image = self._base64_to_image(generated_image_base64)
                    
                    logger.info("Image generated successfully from Vast.ai API")
                    return generated_image
                    
        except aiohttp.ClientError as e:
            logger.error(f"Network error connecting to Vast.ai API: {e}")
            raise Exception(f"Failed to connect to Vast.ai API: {str(e)}")
        except Exception as e:
            logger.error(f"Error in Vast.ai API call: {e}")
            raise
