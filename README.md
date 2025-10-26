# MyWork

![mujoco-deployment](imgs/mujoco_deployment.gif)


## Installation and Execution Guide

### 1. Clone Repository

```bash
# Install Git LFS if not already installed
sudo apt update && sudo apt install git-lfs -y

# Initialize Git LFS
git lfs install

# Clone repository (including LFS files)
git clone https://github.com/KimDoYoung1997/MyWork.git
cd MyWork
```

### 2. Install Required Packages

```bash
# Create conda environment
conda create -n env_isaaclab python=3.11

# Activate environment
conda activate env_isaaclab

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# Install Isaac Sim
pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com

# Install additional required packages
pip3 install onnx
pip3 install mujoco
pip3 install onnxruntime
```

### 3. Execution

```bash
# Activate conda environment first
conda activate env_isaaclab

# Use all options
python3 main.py --motion_file dance2_subject5 --policy_file dance2_subject5 --duration 30.0

# Show help
python3 main.py --help
```

#### Available Options:
- `--motion_file`: Motion file name (without extension, default: dance1_subject1)
- `--policy_file`: Policy file name (without extension, default: dance1_subject1)  
- `--duration`: Simulation duration in seconds (default: 30.0)

#### Available Motion/Policy Files:
- `dance1_subject1` (default)
- `dance2_subject5`


## Project Structure

```
MyWork/
├── config/                # Configuration files
├── modules/               # Python modules
├── npzs/                  # Motion data files
├── performance_plots/     # Performance analysis plots
├── policies/              # Trained policy files
├── unitree_description/   # Unitree robot files
└── main.py     # Unified main script
```
## Reference
- [Whole Body Tracking](https://github.com/HybridRobotics/whole_body_tracking.git)
- [Unitree MuJoCo](https://github.com/unitreerobotics/unitree_mujoco)