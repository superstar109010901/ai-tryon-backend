"""
Model Manager for SDXL Models
Handles downloading, loading, and managing SDXL models from HuggingFace.
Supports automatic model download if models are not present.
"""

import os
import torch
from pathlib import Path
from typing import Optional, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages SDXL models: downloading, loading, and providing access to them.
    
    Models managed:
    - SDXL Base (for image generation)
    - SDXL Inpainting (for clothing replacement)
    - SDXL Refiner (optional, for higher quality)
    """
    
    def __init__(self, models_dir: str = "models", gpu_id: int = 0):
        """
        Initialize ModelManager with directory for storing models.
        Optimized for GPU server deployment.
        
        Args:
            models_dir: Directory where models will be stored
            gpu_id: GPU device ID to use (default: 0, for multi-GPU setups)
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Model paths and configurations
        self.model_configs = {
            "sdxl_base": {
                "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
                "local_dir": self.models_dir / "sdxl-base",
                "loaded": False,
                "pipeline": None
            },
            "sdxl_inpainting": {
                "repo_id": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                "local_dir": self.models_dir / "sdxl-inpainting",
                "loaded": False,
                "pipeline": None
            },
            "sdxl_refiner": {
                "repo_id": "stabilityai/stable-diffusion-xl-refiner-1.0",
                "local_dir": self.models_dir / "sdxl-refiner",
                "loaded": False,
                "pipeline": None,
                "optional": True  # Refiner is optional for faster processing
            }
        }
        
        # Device configuration (GPU if available, else CPU)
        # For GPU servers: Use specific GPU device
        if torch.cuda.is_available():
            self.device = f"cuda:{gpu_id}" if torch.cuda.device_count() > 1 else "cuda"
            self.dtype = torch.float16  # Use float16 for GPU (faster, less memory)
            
            # Log GPU information
            gpu_name = torch.cuda.get_device_name(gpu_id)
            gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
            logger.info(f"GPU Server Detected: {gpu_name}")
            logger.info(f"GPU Memory: {gpu_memory:.2f} GB")
            logger.info(f"Using device: {self.device}")
        else:
            self.device = "cpu"
            self.dtype = torch.float32
            logger.warning("No GPU detected! Running on CPU (will be very slow)")
        
        logger.info(f"Initialized ModelManager with device: {self.device}, dtype: {self.dtype}")
    
    def check_gpu_available(self) -> bool:
        """Check if GPU is available for processing"""
        return torch.cuda.is_available()
    
    async def ensure_models_available(self):
        """
        Ensure all required models are downloaded and available.
        Downloads models from HuggingFace if they don't exist locally.
        """
        try:
            from diffusers import DiffusionPipeline, StableDiffusionXLInpaintPipeline
            from huggingface_hub import snapshot_download
            
            logger.info("Checking model availability...")
            
            # Check and download base SDXL model
            if not self.model_configs["sdxl_base"]["local_dir"].exists():
                logger.info("Downloading SDXL base model...")
                snapshot_download(
                    repo_id=self.model_configs["sdxl_base"]["repo_id"],
                    local_dir=str(self.model_configs["sdxl_base"]["local_dir"]),
                    local_dir_use_symlinks=False
                )
            
            # Check and download inpainting model
            if not self.model_configs["sdxl_inpainting"]["local_dir"].exists():
                logger.info("Downloading SDXL inpainting model...")
                snapshot_download(
                    repo_id=self.model_configs["sdxl_inpainting"]["repo_id"],
                    local_dir=str(self.model_configs["sdxl_inpainting"]["local_dir"]),
                    local_dir_use_symlinks=False
                )
            
            # Optional: Download refiner (commented out by default for faster startup)
            # Uncomment if you want higher quality results
            # if not self.model_configs["sdxl_refiner"]["local_dir"].exists():
            #     logger.info("Downloading SDXL refiner model...")
            #     snapshot_download(
            #         repo_id=self.model_configs["sdxl_refiner"]["repo_id"],
            #         local_dir=str(self.model_configs["sdxl_refiner"]["local_dir"]),
            #         local_dir_use_symlinks=False
            #     )
            
            logger.info("Models are available!")
            
        except ImportError:
            logger.warning(
                "diffusers or huggingface_hub not installed. "
                "Install with: pip install diffusers transformers accelerate huggingface_hub"
            )
            raise
        except Exception as e:
            logger.error(f"Error ensuring models are available: {e}")
            raise
    
    def load_models(self):
        """
        Load SDXL models into memory.
        This is called when we need to actually use the models for generation.
        """
        try:
            from diffusers import DiffusionPipeline, StableDiffusionXLInpaintPipeline
            
            # Load inpainting pipeline (main model we'll use)
            if not self.model_configs["sdxl_inpainting"]["loaded"]:
                logger.info("Loading SDXL inpainting model...")
                pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
                    str(self.model_configs["sdxl_inpainting"]["local_dir"]),
                    torch_dtype=self.dtype,
                    use_safetensors=True
                )
                pipeline = pipeline.to(self.device)
                
                # GPU optimization settings for dedicated GPU servers
                # Enable memory efficient attention to reduce VRAM usage
                if hasattr(pipeline, "enable_attention_slicing"):
                    pipeline.enable_attention_slicing()
                
                # For GPU servers: Keep models on GPU (don't offload to CPU)
                # CPU offloading is disabled for better GPU performance
                # Only use if you have very limited GPU memory (<8GB VRAM)
                # if hasattr(pipeline, "enable_model_cpu_offload"):
                #     pipeline.enable_model_cpu_offload()
                
                # Enable VAE slicing for additional memory savings if available
                if hasattr(pipeline, "enable_vae_slicing"):
                    pipeline.enable_vae_slicing()
                
                self.model_configs["sdxl_inpainting"]["pipeline"] = pipeline
                self.model_configs["sdxl_inpainting"]["loaded"] = True
                
                # Log GPU memory info if available
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    logger.info(f"SDXL inpainting model loaded on {self.device}!")
                    logger.info(f"GPU Memory: {gpu_memory:.2f} GB available")
                else:
                    logger.info("SDXL inpainting model loaded on CPU!")
            
            return self.model_configs["sdxl_inpainting"]["pipeline"]
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def get_inpainting_pipeline(self):
        """
        Get the loaded inpainting pipeline.
        Loads it if not already loaded.
        """
        if not self.model_configs["sdxl_inpainting"]["loaded"]:
            return self.load_models()
        return self.model_configs["sdxl_inpainting"]["pipeline"]
    
    def unload_models(self):
        """
        Unload models from memory to free up GPU memory.
        Useful when not processing images.
        """
        for model_name, config in self.model_configs.items():
            if config["loaded"] and config["pipeline"] is not None:
                del config["pipeline"]
                config["pipeline"] = None
                config["loaded"] = False
                logger.info(f"Unloaded {model_name}")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


