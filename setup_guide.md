# Setup Guide

This guide will help you set up the Real-Time Bag Counter system on both PC and Raspberry Pi platforms.

## 📋 Prerequisites

### System Requirements

**For PC:**
- Windows 10/11, macOS 10.14+, or Linux Ubuntu 18.04+
- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- USB camera or webcam
- 2GB free disk space

**For Raspberry Pi:**
- Raspberry Pi 4B or 5 (8GB RAM recommended)
- Raspberry Pi OS (64-bit recommended)
- Python 3.8+
- Camera module or USB camera
- 32GB+ microSD card (Class 10)

## 🛠️ Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/bag-counter.git
cd bag-counter
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv bag_counter_env

# Activate on Windows
bag_counter_env\Scripts\activate

# Activate on macOS/Linux
source bag_counter_env/bin/activate
```

### 3. Install Dependencies

```bash
# Install core requirements
pip install -r requirements.txt

# For GPU support (optional)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. Download or Train Model

**Option A: Use Pre-trained Model**
- Place your trained `best.pt` model in the `models/` directory
- Update the model path in `config.yaml`

**Option B: Train Your Own Model**
1. Prepare your dataset in YOLO format
2. Update class names in `data/classes.txt`
3. Run the training notebook: `jupyter notebook training_notebook.ipynb`

### 5. Configure Settings

Edit `config.yaml` to match your setup:

```yaml
model:
  path: "models/best.pt"  # Update this path
  
camera:
  index: 0  # Your camera index
  width: 1280  # Adjust based on your hardware
  height: 720
```

## 🖥️ PC Setup

### Additional PC Requirements

```bash
# Windows users may need Visual C++ redistributables
# Install from: https://aka.ms/vs/17/release/vc_redist.x64.exe

# For better camera support on Windows
pip install opencv-python-headless
```

### Running on PC

```bash
# Run with default settings
python src/pc_bag_counter.py

# Run diagnostics first
python src/pc_bag_counter.py --diagnostics

# Custom configuration
python src/pc_bag_counter.py --config custom_config.yaml
```

## 🥧 Raspberry Pi Setup

### Raspberry Pi Specific Installation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3-pip python3-venv libopencv-dev
sudo apt install -y libatlas-base-dev libhdf5-dev libhdf5-serial-dev
sudo apt install -y libjpeg-dev libtiff5-dev libpng-dev
sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev
sudo apt install -y libgtk2.0-dev libcanberra-gtk-module libcanberra-gtk3-module

# Enable camera (for Pi Camera module)
sudo raspi-config
# Navigate to: Interfacing Options > Camera > Enable
```

### Optimize for Raspberry Pi

```bash
# Increase swap space (optional but recommended)
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Change CONF_SWAPSIZE=100 to CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Install lightweight PyTorch
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
```

### Running on Raspberry Pi

```bash
# Run diagnostics first
python src/realtime_bag_counter_pi.py --diagnostics

# Run main application
python src/realtime_bag_counter_pi.py
```

## 📷 Camera Setup

### USB Camera
1. Connect USB camera
2. Test camera access:
   ```bash
   # List available cameras
   ls /dev/video*
   
   # Test with OpenCV
   python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera works:', cap.isOpened()); cap.release()"
   ```

### Raspberry Pi Camera Module
1. Connect camera module to CSI port
2. Enable camera in `raspi-config`
3. Test camera:
   ```bash
   # Test with libcamera (newer Pi OS)
   libcamera-hello --timeout 5000
   
   # Or with legacy tools
   raspistill -o test.jpg
   ```

## 🔧 Configuration

### Hardware Profile Selection

The system automatically detects your platform, but you can override:

```python
# In your config.yaml
active_profile: "pc"  # or "raspberry_pi"
```

### Camera Configuration

```yaml
camera:
  index: 0  # Try different values (0, 1, 2) if camera not found
  backend: "v4l2"  # Linux: v4l2, Windows: dshow, Auto: auto
```

### Performance Tuning

**For slower hardware:**
```yaml
performance:
  skip_frames: 3  # Process every 3rd frame
  max_detections: 30
camera:
  width: 480
  height: 320
```

**For powerful hardware:**
```yaml
performance:
  skip_frames: 1  # Process every frame
  max_detections: 200
camera:
  width: 1920
  height: 1080
```

## 🧪 Testing Installation

### Run Diagnostics

```bash
# Comprehensive system check
python src/pc_bag_counter.py --diagnostics

# Or for Raspberry Pi
python src/realtime_bag_counter_pi.py --diagnostics
```

### Verify Components

```python
# Test script to verify installation
python -c "
import cv2
import torch
import ultralytics
print('OpenCV version:', cv2.__version__)
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('Ultralytics version:', ultralytics.__version__)
"
```

## 🐛 Troubleshooting

### Common Issues

**Model Loading Error:**
```bash
# Verify model file exists
ls -la models/best.pt

# Check model compatibility
python -c "from ultralytics import YOLO; model = YOLO('models/best.pt'); print('Model loaded successfully')"
```

**Camera Not Found:**
```bash
# List all video devices
ls /dev/video*

# Test different camera indices
python -c "
import cv2
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'Camera {i}: Available')
        cap.release()
    else:
        print(f'Camera {i}: Not available')
"
```

**Performance Issues:**
- Reduce frame resolution
- Increase `skip_frames` value
- Lower `max_detections` limit
- Ensure adequate cooling for Raspberry Pi

**Permission Errors (Linux):**
```bash
# Add user to video group
sudo usermod -a -G video $USER

# Set camera permissions
sudo chmod 666 /dev/video0
```

## 🚀 Optimization Tips

### For Best Performance

1. **Use dedicated GPU** (if available)
2. **Optimize camera settings** for your lighting conditions
3. **Adjust confidence thresholds** based on your accuracy requirements
4. **Use appropriate frame rates** for your hardware
5. **Enable cooling** for continuous operation

### Memory Management

```python
# Add to your configuration for low-memory systems
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
```

## 📊 Performance Benchmarks

| Hardware | Resolution | FPS | Processing Time |
|----------|------------|-----|-----------------|
| PC (RTX 3060) | 1280x720 | 30+ | 15-25ms |
| PC (CPU only) | 1280x720 | 15-20 | 50-80ms |
| Pi 5 (8GB) | 640x480 | 10-15 | 100-150ms |
| Pi 4 (8GB) | 640x480 | 8-12 | 150-200ms |

## 📞 Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Run diagnostics: `--diagnostics` flag
3. Review logs in the console output
4. Open an issue on GitHub with:
   - Your hardware specifications
   - Error messages
   - Configuration file
   - Diagnostic output

## 🔄 Updates

Keep your installation updated:

```bash
# Update the repository
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Update PyTorch (if needed)
pip install torch torchvision --upgrade
```