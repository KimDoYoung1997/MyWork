# MyWork

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
# TODO

```

### 3. Execution

```bash
# Use all options
python3 my_code_unified.py --motion_file dance2_subject5 --policy_file dance2_subject5 --duration 30.0

# Show help
python3 my_code_unified.py --help
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
└── my_code_unified.py     # Unified main script
```
