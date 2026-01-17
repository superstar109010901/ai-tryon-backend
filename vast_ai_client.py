"""
Vast.ai Stable Diffusion API Client
Handles communication with Vast.ai's Stable Diffusion API for img2img generation.
"""

import base64
import io
import asyncio
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
    
    def __init__(self, api_url: str, timeout: int = 600):
        """
        Initialize Vast.ai API client.
        
        Args:
            api_url: Base URL of Vast.ai SD API (e.g., "http://localhost:8081")
            timeout: Request timeout in seconds (default: 600 = 10 minutes for image generation)
        """
        self.api_url = api_url.rstrip('/')
        self.img2img_endpoint = f"{self.api_url}/sdapi/v1/img2img"
        self.controlnet_detect_endpoint = f"{self.api_url}/controlnet/detect"  # For segmentation preprocessing
        self.timeout = timeout
        self.max_retries = 2  # Retry up to 2 times for transient failures
        logger.info(f"Initialized Vast.ai client with API URL: {self.api_url}")
        logger.info(f"ControlNet detect endpoint: {self.controlnet_detect_endpoint}")
        logger.info(f"Timeout: {self.timeout}s, Max retries: {self.max_retries}")
    
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
    
    async def detect_face_area(self, image: Image.Image) -> Optional[Image.Image]:
        """
        Detect face area using ControlNet face detection.
        Uses mediapipe_face or openpose_full to detect face/head region.
        
        Args:
            image: Input PIL Image
        
        Returns:
            Face detection map as PIL Image, or None if detection fails
        """
        try:
            image_base64 = self._image_to_base64(image)
            
            # Try mediapipe_face first (better for face detection)
            # Fallback to openpose_full if mediapipe_face not available
            modules_to_try = ["mediapipe_face", "openpose_full", "openpose"]
            
            for module_name in modules_to_try:
                try:
                    payload = {
                        "controlnet_module": module_name,
                        "controlnet_input_images": [image_base64],
                        "controlnet_processor_res": 512,
                    }
                    
                    logger.info(f"Requesting face detection using {module_name}...")
                    
                    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                        async with session.post(
                            self.controlnet_detect_endpoint,
                            json=payload,
                            headers={"Content-Type": "application/json"}
                        ) as response:
                            if response.status == 200:
                                result = await response.json()
                                
                                if "images" not in result or len(result["images"]) == 0:
                                    logger.warning(f"No detection image returned from {module_name}")
                                    continue
                                
                                detection_base64 = result["images"][0]
                                detection_image = self._base64_to_image(detection_base64)
                                
                                logger.info(f"Face detection successful using {module_name}")
                                return detection_image
                            else:
                                logger.warning(f"{module_name} returned status {response.status}, trying next...")
                                continue
                
                except Exception as e:
                    logger.warning(f"Error with {module_name}: {e}, trying next...")
                    continue
            
            logger.warning("All face detection methods failed, falling back to percentage-based protection")
            return None
                    
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return None
    
    async def detect_person_and_clothing(self, image: Image.Image) -> Dict[str, Optional[Image.Image]]:
        """
        Detect full person and clothing regions using ControlNet segmentation.
        Returns face detection and segmentation map for clothing detection.
        
        Args:
            image: Input PIL Image
        
        Returns:
            Dictionary with 'face' and 'segmentation' keys, values are PIL Images or None
        """
        result = {
            'face': None,
            'segmentation': None
        }
        
        # Detect face for protection
        try:
            result['face'] = await self.detect_face_area(image)
            if result['face'] is not None:
                logger.info("✅ Face detection successful")
        except Exception as e:
            logger.warning(f"Face detection error: {e}")
        
        # Detect full person and clothing using segmentation
        try:
            result['segmentation'] = await self.get_segmentation_map(image)
            if result['segmentation'] is not None:
                logger.info("✅ Person/body segmentation successful")
        except Exception as e:
            logger.warning(f"Segmentation error: {e}")
        
        return result
    
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
        denoising_strength: float = 0.30,
        steps: int = 25,
        cfg_scale: float = 5,
        sampler_name: str = "DPM++ SDE",
        width: int = 1024,
        height: int = 1024,
        controlnet_pose_enabled: bool = True,
        controlnet_pose_module: Optional[str] = "openpose_full",
        controlnet_pose_model: Optional[str] = "controlnet-openpose-sdxl",
        controlnet_pose_weight: float = 1.0,
        controlnet_pose_control_mode: str = "ControlNet is more important",
        controlnet_inpaint_enabled: bool = True,
        controlnet_inpaint_model: Optional[str] = "controlnet-inpaint-sdxl",
        controlnet_inpaint_weight: float = 1.0,
        controlnet_pixel_perfect: bool = True
    ) -> Image.Image:
        """
        Generate image using Vast.ai img2img API with inpainting mask + dual ControlNet.
        
        Uses img2img + two ControlNet units + Mask for precise clothing replacement:
        - Unit 0: OpenPose for pose lock (preserves full body)
        - Unit 1: Inpaint for inpainting guidance (helps with clothing replacement)
        
        Args:
            image: Input PIL Image
            prompt: Positive prompt for generation
            negative_prompt: Negative prompt (what to avoid)
            mask: Mask image for inpainting (white=change, black=preserve)
            denoising_strength: How much to change the image (0.25-0.35 recommended)
            steps: Number of inference steps
            cfg_scale: Guidance scale (how closely to follow prompt)
            sampler_name: Sampler to use (e.g., "DPM++ SDE")
            width: Output image width
            height: Output image height
            controlnet_pose_enabled: Whether to use OpenPose ControlNet (Unit 0)
            controlnet_pose_module: OpenPose module ("openpose_full")
            controlnet_pose_model: OpenPose model ("controlnet-openpose-sdxl")
            controlnet_pose_weight: OpenPose weight (1.0 recommended)
            controlnet_pose_control_mode: OpenPose control mode ("ControlNet is more important")
            controlnet_inpaint_enabled: Whether to use Inpaint ControlNet (Unit 1)
            controlnet_inpaint_model: Inpaint model ("controlnet-inpaint-sdxl")
            controlnet_inpaint_weight: Inpaint weight (1.0 recommended)
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
            
            # Build payload matching exact format for img2img INPAINT
            # Must include inpainting parameters for proper inpainting mode
            # CRITICAL: Use inpainting_fill: 0 (latent noise) for natural blending, not 1 (original)
            # inpainting_fill: 0 = better blending, 1 = can look pasted/overlaid
            # TRUE TRY-ON: Garment replacement (not pure recolor)
            # For true try-on with SDXL: masked_content = latent_noise allows full garment replacement
            # For texture-space recolor (color-only): SDXL would NOT be used - would use UV/texture-space approach
            # We're doing garment replacement, so latent_noise is appropriate
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
                "inpainting_fill": 0,  # 0 = latent_noise (allows clothes to change, was 1 which preserved too much)
                "inpaint_full_res": True,  # True for better quality
                "inpaint_full_res_padding": 24,  # Moderate padding for better blending
                "inpaint_area": 1,  # 1 = only masked area, 0 = whole picture
                "mask_blur": 4,  # Moderate blur for smooth edges
            }
            
            # Add ControlNet using controlnet_units format with TWO units:
            # Unit 0: OpenPose for pose lock (preserves full body pose)
            # Unit 1: Inpaint for inpainting guidance (helps with clothing replacement)
            controlnet_units = []
            
            # ControlNet Unit 0: Pose lock (openpose_full)
            if controlnet_pose_enabled and controlnet_pose_model:
                controlnet_units.append({
                    "enabled": True,
                    "image": image_base64,  # Use original image for pose detection
                    "module": controlnet_pose_module if controlnet_pose_module else "openpose_full",
                    "model": controlnet_pose_model,
                    "weight": controlnet_pose_weight,
                    "pixel_perfect": controlnet_pixel_perfect,
                    "control_mode": controlnet_pose_control_mode
                })
                logger.info(f"ControlNet Unit 0 (Pose lock): {controlnet_pose_model} / {controlnet_pose_module}, weight={controlnet_pose_weight}, mode={controlnet_pose_control_mode}")
            
            # ControlNet Unit 1: Inpaint guidance
            # FIX 3: ControlNet mode correction - prevents OpenPose from overpowering inpaint
            if controlnet_inpaint_enabled and controlnet_inpaint_model:
                # Ensure weight is in range 0.4-0.6
                inpaint_weight = max(0.4, min(0.6, controlnet_inpaint_weight))
                controlnet_units.append({
                    "enabled": True,
                    "image": image_base64,  # Use original image for inpainting guidance
                    "module": "inpaint",  # Inpaint preprocessor
                    "model": controlnet_inpaint_model,
                    "weight": inpaint_weight,  # FIX 3: Ensure 0.4-0.6 range
                    "pixel_perfect": controlnet_pixel_perfect,
                    "control_mode": "ControlNet is more important",  # FIX 3: Changed from "Balanced"
                    "guidance_start": 0.0,
                    "guidance_end": 0.8  # FIX 3: End at 0.8 (mandatory)
                })
                logger.info(f"ControlNet Unit 1 (Inpaint guidance): {controlnet_inpaint_model}, weight={inpaint_weight}, mode=ControlNet is more important, guidance_end=0.8")
            
            if controlnet_units:
                payload["controlnet_units"] = controlnet_units
                logger.info(f"✅ Added {len(controlnet_units)} ControlNet unit(s)")
            else:
                logger.warning("⚠️  No ControlNet units enabled")
            
            # Send request to Vast.ai API with retry mechanism
            logger.info(f"Sending img2img request to {self.img2img_endpoint}")
            logger.info(f"Prompt: {prompt[:100]}...")
            logger.info(f"Parameters: steps={steps}, denoising_strength={denoising_strength}, cfg_scale={cfg_scale}")
            
            # Retry mechanism for transient failures
            last_error = None
            for attempt in range(self.max_retries + 1):
                try:
                    if attempt > 0:
                        logger.info(f"Retry attempt {attempt}/{self.max_retries}...")
                        await asyncio.sleep(2 * attempt)  # Exponential backoff
                    
                    # Create timeout with separate connect and read timeouts
                    timeout = aiohttp.ClientTimeout(
                        total=self.timeout,
                        connect=30,  # 30s to establish connection
                        sock_read=self.timeout  # Full timeout for reading response
                    )
                    
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.post(
                            self.img2img_endpoint,
                            json=payload,
                            headers={"Content-Type": "application/json"}
                        ) as response:
                            # Check response status
                            if response.status == 200:
                                try:
                                    result = await response.json()
                                except Exception as json_error:
                                    error_text = await response.text()
                                    logger.error(f"Failed to parse JSON response: {json_error}")
                                    logger.error(f"Response text: {error_text[:500]}")
                                    raise Exception(f"Invalid JSON response from API: {str(json_error)}")
                                
                                # Extract generated image from response
                                if "images" not in result:
                                    logger.error(f"Response missing 'images' key. Keys: {list(result.keys())}")
                                    raise Exception("Response missing 'images' field")
                                
                                if len(result["images"]) == 0:
                                    logger.error("Response 'images' array is empty")
                                    raise Exception("No images returned from Vast.ai API")
                                
                                # Convert base64 response to PIL Image
                                try:
                                    generated_image_base64 = result["images"][0]
                                    generated_image = self._base64_to_image(generated_image_base64)
                                    logger.info(f"✅ Image generated successfully from Vast.ai API (attempt {attempt + 1})")
                                    return generated_image
                                except Exception as img_error:
                                    logger.error(f"Failed to decode image from base64: {img_error}")
                                    raise Exception(f"Failed to decode generated image: {str(img_error)}")
                            
                            elif response.status == 503 or response.status == 502:
                                # Service unavailable or bad gateway - retry
                                error_text = await response.text()
                                logger.warning(f"API returned {response.status} (attempt {attempt + 1}/{self.max_retries + 1}): {error_text[:200]}")
                                last_error = f"Vast.ai API temporarily unavailable ({response.status}): {error_text[:200]}"
                                if attempt < self.max_retries:
                                    continue  # Retry
                                else:
                                    raise Exception(last_error)
                            
                            else:
                                # Other errors - don't retry
                                error_text = await response.text()
                                error_msg = f"Vast.ai API error {response.status}: {error_text[:500]}"
                                logger.error(f"API Request failed:")
                                logger.error(f"  URL: {self.img2img_endpoint}")
                                logger.error(f"  Status: {response.status}")
                                logger.error(f"  Error: {error_text[:500]}")
                                raise Exception(error_msg)
                
                except asyncio.TimeoutError as e:
                    last_error = f"Request timeout after {self.timeout}s (attempt {attempt + 1}/{self.max_retries + 1})"
                    logger.warning(f"{last_error}")
                    if attempt < self.max_retries:
                        continue  # Retry
                    else:
                        raise Exception(f"Request timeout: API did not respond within {self.timeout} seconds. The image generation may be taking too long.")
                
                except aiohttp.ClientConnectorError as e:
                    last_error = f"Connection error: {str(e)}"
                    logger.warning(f"{last_error} (attempt {attempt + 1}/{self.max_retries + 1})")
                    if attempt < self.max_retries:
                        continue  # Retry
                    else:
                        raise Exception(f"Failed to connect to Vast.ai API: {str(e)}. Check if the API server is running at {self.api_url}")
                
                except aiohttp.ServerDisconnectedError as e:
                    last_error = f"Server disconnected: {str(e)}"
                    logger.warning(f"{last_error} (attempt {attempt + 1}/{self.max_retries + 1})")
                    if attempt < self.max_retries:
                        continue  # Retry
                    else:
                        raise Exception(f"Server disconnected during request: {str(e)}")
            
            # If we get here, all retries failed
            raise Exception(f"All retry attempts failed. Last error: {last_error}")
                    
        except aiohttp.ClientError as e:
            logger.error(f"Network error connecting to Vast.ai API: {e}")
            raise Exception(f"Network error: Failed to connect to Vast.ai API: {str(e)}")
        except Exception as e:
            logger.error(f"Error in Vast.ai API call: {e}")
            raise
