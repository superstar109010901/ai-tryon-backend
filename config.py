"""
Configuration file for Vast.ai API connection
"""

import os

# Vast.ai Stable Diffusion API URL
# For Vast.ai: Use the external port that maps to SD API
# Your Vast.ai server: SD WebUI runs on external port 8081
#   VAST_AI_SD_URL = "http://74.48.140.178:8081"  # SD API external port on Vast.ai
VAST_AI_SD_URL = os.getenv("VAST_AI_SD_URL", "http://localhost:8081")

# ControlNet Configuration
# ControlNet with OpenPose is used to preserve pose and body structure
CONTROLNET_ENABLED = os.getenv("CONTROLNET_ENABLED", "true").lower() == "true"  # Enabled by default
CONTROLNET_MODULE = os.getenv("CONTROLNET_MODULE", "openpose")  # OpenPose for pose preservation
CONTROLNET_MODEL = os.getenv("CONTROLNET_MODEL", "control_sd15_openpose")  # SD15 OpenPose model
CONTROLNET_WEIGHT = float(os.getenv("CONTROLNET_WEIGHT", "1.0"))  # Locks pose
CONTROLNET_GUIDANCE_START = float(os.getenv("CONTROLNET_GUIDANCE_START", "0.0"))
CONTROLNET_GUIDANCE_END = float(os.getenv("CONTROLNET_GUIDANCE_END", "0.9"))
CONTROLNET_CONTROL_MODE = os.getenv("CONTROLNET_CONTROL_MODE", "Balanced")
