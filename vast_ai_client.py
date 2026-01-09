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
        self.timeout = timeout
        logger.info(f"Initialized Vast.ai client with API URL: {self.api_url}")
    
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
    
    async def generate_img2img(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        mask: Optional[Image.Image] = None,
        denoising_strength: float = 0.32,
        steps: int = 25,
        cfg_scale: float = 7.0,
        sampler_name: str = "DPM++ 2M Karras",
        width: int = 512,
        height: int = 768,
        inpainting_fill: int = 1,
        inpaint_full_res: bool = True,
        inpaint_full_res_padding: int = 32,
        controlnet_enabled: bool = False,
        controlnet_module: Optional[str] = None,
        controlnet_model: Optional[str] = None,
        controlnet_weight: float = 1.0
    ) -> Image.Image:
        """
        Generate image using Vast.ai img2img API with optional inpainting mask.
        
        Args:
            image: Input PIL Image
            prompt: Positive prompt for generation
            negative_prompt: Negative prompt (what to avoid)
            mask: Optional mask image for inpainting (white=change, black=preserve)
            denoising_strength: How much to change the image (0.0-1.0)
            steps: Number of inference steps
            cfg_scale: Guidance scale (how closely to follow prompt)
            sampler_name: Sampler to use (e.g., "DPM++ 2M Karras")
            width: Output image width
            height: Output image height
            inpainting_fill: Inpainting fill mode (0=fill, 1=original, 2=latent noise, 3=latent nothing)
            inpaint_full_res: Use full resolution inpainting
            inpaint_full_res_padding: Padding for full resolution inpainting
            controlnet_enabled: Whether to use ControlNet
            controlnet_module: ControlNet module name (e.g., "openpose")
            controlnet_model: ControlNet model name (e.g., "control_v11p_sd15_openpose")
            controlnet_weight: ControlNet weight (0.0-2.0)
        
        Returns:
            Generated PIL Image
        """
        try:
            # Convert image to base64
            image_base64 = self._image_to_base64(image)
            
            # Build payload
            payload = {
                "init_images": [image_base64],
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "denoising_strength": denoising_strength,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "sampler_name": sampler_name,
                "width": width,
                "height": height
            }
            
            # Add mask for inpainting if provided
            if mask is not None:
                mask_base64 = self._image_to_base64(mask)
                payload["mask"] = mask_base64
                payload["inpainting_fill"] = inpainting_fill
                payload["inpaint_full_res"] = inpaint_full_res
                payload["inpaint_full_res_padding"] = inpaint_full_res_padding
                logger.info("Using inpainting with mask (white=change, black=preserve)")
            
            # Add ControlNet if enabled
            if controlnet_enabled and controlnet_module and controlnet_model:
                payload["alwayson_scripts"] = {
                    "controlnet": {
                        "args": [{
                            "input_image": image_base64,
                            "module": controlnet_module,
                            "model": controlnet_model,
                            "weight": controlnet_weight
                        }]
                    }
                }
                logger.info(f"ControlNet enabled: {controlnet_module} / {controlnet_model}")
            
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
                        raise Exception(f"Vast.ai API error {response.status}: {error_text}")
                    
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
