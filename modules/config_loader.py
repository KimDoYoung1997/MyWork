"""
Configuration loader for robot body and joint definitions
This module provides functions to load robot configuration from YAML files
"""

import yaml
import os
from typing import List, Dict, Any

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration data
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    return config

def get_mujoco_joint_sequence(config_dir: str = None) -> List[str]:
    """Load MuJoCo joint sequence from configuration file
    
    MuJoCo XML 파일(g1.xml)에 정의된 관절 순서를 로드합니다.
    Isaac Lab과 MuJoCo 간 관절 순서가 다르므로 이 매핑이 필요합니다.
    
    Args:
        config_dir: Directory containing configuration files (optional)
                    None이면 현재 파일의 상위 디렉토리의 config 폴더 사용
        
    Returns:
        List of joint names in MuJoCo order (29개 관절)
    """
    if config_dir is None:
        # Default to config directory relative to this module
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_dir = os.path.join(script_dir, 'config')
    
    config_path = os.path.join(config_dir, 'mujoco_joint_sequence.yaml')
    config = load_yaml_config(config_path)
    
    return config['mujoco_joint_sequence']

def get_isaac_body_names(config_dir: str = None) -> List[str]:
    """Load Isaac Lab body names from configuration file
    
    Isaac Lab에서 사용하는 body 이름 순서를 로드합니다.
    모션 데이터(NPZ)의 body_pos_w, body_quat_w 배열이 이 순서를 따릅니다.
    
    Args:
        config_dir: Directory containing configuration files (optional)
                    None이면 현재 파일의 상위 디렉토리의 config 폴더 사용
        
    Returns:
        List of body names in Isaac Lab order (30개 body)
        인덱스 9 = 'torso_link' (anchor body)
    """
    if config_dir is None:
        # Default to config directory relative to this module
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_dir = os.path.join(script_dir, 'config')
    
    config_path = os.path.join(config_dir, 'isaac_body_names.yaml')
    config = load_yaml_config(config_path)
    
    return config['isaac_body_names']

def get_anchor_body_info(config_dir: str = None) -> Dict[str, Any]:
    """Load anchor body information from Isaac Lab configuration
    
    Args:
        config_dir: Directory containing configuration files (optional)
        
    Returns:
        Dictionary containing anchor body information
    """
    if config_dir is None:
        # Default to scripts/config directory
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_dir = os.path.join(script_dir, 'config')
    
    config_path = os.path.join(config_dir, 'isaac_body_names.yaml')
    config = load_yaml_config(config_path)
    
    return config['body_mapping']['anchor_body']

def get_joint_groups(config_dir: str = None) -> Dict[str, List[str]]:
    """Load joint groups from MuJoCo configuration
    
    Args:
        config_dir: Directory containing configuration files (optional)
        
    Returns:
        Dictionary mapping group names to joint lists
    """
    if config_dir is None:
        # Default to scripts/config directory
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_dir = os.path.join(script_dir, 'config')
    
    config_path = os.path.join(config_dir, 'mujoco_joint_sequence.yaml')
    config = load_yaml_config(config_path)
    
    return config['joint_mapping']['joint_groups']

def get_body_groups(config_dir: str = None) -> Dict[str, List[str]]:
    """Load body groups from Isaac Lab configuration
    
    Args:
        config_dir: Directory containing configuration files (optional)
        
    Returns:
        Dictionary mapping group names to body lists
    """
    if config_dir is None:
        # Default to scripts/config directory
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_dir = os.path.join(script_dir, 'config')
    
    config_path = os.path.join(config_dir, 'isaac_body_names.yaml')
    config = load_yaml_config(config_path)
    
    return config['body_mapping']['body_groups']
