"""
Configuration file for Vast.ai API connection
"""

import os

# Vast.ai Stable Diffusion API URL
# IMPORTANT: Choose the correct URL based on where your backend runs:
#
# Option 1: Backend and SD WebUI on SAME Vast.ai server (RECOMMENDED)
#   Use internal port: http://localhost:7860
#   (SD WebUI runs on internal port 7860, maps to external 8081)
#
# Option 2: Backend on different server (or local machine)
#   Use external port: http://74.48.140.178:8081
#   (Access SD WebUI from outside via external port 8081)
#
# Set via environment variable:
#   export VAST_AI_SD_URL="http://localhost:7860"  # Same server (default)
#   export VAST_AI_SD_URL="http://74.48.140.178:8081"  # Remote access
VAST_AI_SD_URL = os.getenv("VAST_AI_SD_URL", "http://localhost:7860")

# Backend URL for serving images
# This is the external URL where the backend is accessible
# Used to construct full image URLs for frontend
# IMPORTANT: This should be the external Vast.ai URL (port 36580 maps to internal 8384)
BACKEND_URL = os.getenv("BACKEND_URL", "http://74.48.140.178:36580")

# ControlNet Configuration
# ControlNet with OpenPose is used to preserve pose and body structure
CONTROLNET_ENABLED = os.getenv("CONTROLNET_ENABLED", "true").lower() == "true"  # Enabled by default
CONTROLNET_MODULE = os.getenv("CONTROLNET_MODULE", "openpose")  # OpenPose for pose preservation
CONTROLNET_MODEL = os.getenv("CONTROLNET_MODEL", "control_sd15_openpose")  # SD15 OpenPose model
CONTROLNET_WEIGHT = float(os.getenv("CONTROLNET_WEIGHT", "1.2"))  # Strong control to preserve person identity and pose
CONTROLNET_GUIDANCE_START = float(os.getenv("CONTROLNET_GUIDANCE_START", "0.0"))
CONTROLNET_GUIDANCE_END = float(os.getenv("CONTROLNET_GUIDANCE_END", "0.9"))
CONTROLNET_CONTROL_MODE = os.getenv("CONTROLNET_CONTROL_MODE", "Balanced")
