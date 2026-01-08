# GPU Server Setup Guide

This guide helps you set up the backend on a GPU server (Vast.ai, RunPod, etc.).

## 🚀 Quick Setup for GPU Servers

### Step 1: Install CUDA-Enabled PyTorch

**IMPORTANT:** The default `requirements.txt` installs CPU-only PyTorch. For GPU servers, you MUST install CUDA-enabled PyTorch first.

```bash
# Check your CUDA version first
nvidia-smi

# Install PyTorch with CUDA 11.8 (most common)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# OR for CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Then install other dependencies
pip install -r requirements.txt
```

### Step 2: Verify GPU Detection

```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
CUDA Available: True
GPU: NVIDIA GeForce RTX 3090
```

### Step 3: Run the Backend

```bash
cd backend
python main.py
```

You should see:
```
============================================================
Initializing Virtual Try-On Backend for GPU Server...
============================================================
GPU Server Detected: NVIDIA GeForce RTX 3090
GPU Memory: 24.00 GB
Using device: cuda
Initialized ModelManager with device: cuda, dtype: torch.float16
...
✅ Models loaded successfully on GPU!
✅ Server ready to process requests
============================================================
```

## 🔧 Optimizations Applied

The backend is now optimized for GPU servers:

1. ✅ **No CPU Offloading** - Models stay on GPU for maximum performance
2. ✅ **Preloading** - Models load to GPU on startup (ready immediately)
3. ✅ **Memory Optimization** - Attention slicing and VAE slicing enabled
4. ✅ **Float16 Precision** - Uses half precision for faster inference
5. ✅ **GPU Device Selection** - Supports multi-GPU setups

## 📊 GPU Requirements

- **Minimum:** 8GB VRAM (with CPU offloading enabled)
- **Recommended:** 16GB+ VRAM (for optimal performance)
- **CUDA:** Version 11.8 or 12.1

## 🐛 Troubleshooting

### Issue: "CUDA not available"

**Solution:**
```bash
# Install CUDA-enabled PyTorch
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "Out of memory" errors

**Solution:** The backend uses memory optimizations, but if you still get OOM:
1. Reduce image resolution in `image_processor.py` (line 71)
2. Reduce `num_inference_steps` in `image_processor.py` (line 170)
3. Enable CPU offload (uncomment line 144 in `model_manager.py`)

### Issue: Models download slowly

**Solution:** Models are large (~7GB each). First run will take time. Subsequent runs use cached models.

## 🌐 Vast.ai Specific Setup

1. **Create Instance:**
   - Select GPU with 16GB+ VRAM
   - Choose Ubuntu/CUDA image
   - Ensure port 8000 is open

2. **SSH into instance:**
   ```bash
   ssh root@<vast-ai-ip>
   ```

3. **Install dependencies:**
   ```bash
   apt-get update
   apt-get install -y python3-pip git
   
   # Install CUDA PyTorch
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   
   # Install other dependencies
   pip install -r requirements.txt
   ```

4. **Run backend:**
   ```bash
   cd backend
   python main.py
   ```

5. **Update frontend API URL:**
   - Set `REACT_APP_API_URL=http://<vast-ai-ip>:8000` in frontend

## 📝 Notes

- Models are automatically downloaded on first run
- Models are cached in `backend/models/` directory
- Server binds to `0.0.0.0:8000` to accept external connections
- Production mode: `reload=False` for better performance

