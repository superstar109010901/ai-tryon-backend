"""
Configuration file for Vast.ai API connection
"""

import os

# Vast.ai Stable Diffusion API URL
# Default: localhost:7860 (common SD webui internal port)
# For Vast.ai: Use the external port that maps to SD API
# Your Vast.ai server: SD WebUI runs on external port 8081
#   VAST_AI_SD_URL = "http://localhost:7860"  # If backend and SD API on same server
#   VAST_AI_SD_URL = "http://74.48.140.178:8081"  # SD API external port on Vast.ai
VAST_AI_SD_URL = os.getenv("VAST_AI_SD_URL", "http://localhost:7860")

# ControlNet Configuration (optional)
CONTROLNET_ENABLED = os.getenv("CONTROLNET_ENABLED", "false").lower() == "true"
CONTROLNET_MODULE = os.getenv("CONTROLNET_MODULE", "openpose")  # e.g., "openpose", "canny", "depth"
CONTROLNET_MODEL = os.getenv("CONTROLNET_MODEL", "control_v11p_sd15_openpose")
CONTROLNET_WEIGHT = float(os.getenv("CONTROLNET_WEIGHT", "1.0"))
