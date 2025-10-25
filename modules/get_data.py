import mujoco


def get_representative_bodies():
    """대표 body들 정의 (다리와 팔)
    
    Returns:
        dict: 대표 body 이름 매핑
    """
    return {
        'left_ankle': 'left_ankle_roll_link',     # 왼발 : 7
        'right_ankle': 'right_ankle_roll_link',   # 오른발 : 13
        'left_hand': 'left_wrist_yaw_link',       # 왼손 : 23
        'right_hand': 'right_wrist_yaw_link'      # 오른손 : 30
    }

def get_mujoco_body_ids(mj_model, representative_bodies):
    """대표 body들의 MuJoCo body ID 찾기 (Plot 시 사용)
    
    Args:
        mj_model: MuJoCo 모델
        representative_bodies: 대표 body 이름 매핑
        
    Returns:
        dict: 대표 body들의 MuJoCo body ID
    """
    representative_body_ids = {}
    for key, body_name in representative_bodies.items():
        body_id_rep = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id_rep != -1:
            representative_body_ids[key] = body_id_rep
            print(f"Found {key} ({body_name}): body_id = {body_id_rep}")
        else:
            print(f"Warning: {key} ({body_name}) not found in model")
    return representative_body_ids

def get_isaac_body_indices(isaac_body_names, representative_bodies):
    """대표 body들의 Isaac Lab 인덱스 찾기 (Plot 시 사용)
    
    Args:
        isaac_body_names: Isaac Lab body 이름 리스트
        representative_bodies: 대표 body 이름 매핑
        
    Returns:
        dict: 대표 body들의 Isaac Lab 인덱스
    """
    isaac_representative_body_ids = {}
    for key, body_name in representative_bodies.items():
        if body_name in isaac_body_names:
            isaac_representative_body_ids[key] = isaac_body_names.index(body_name)
        else:
            print(f"Warning: {key} ({body_name}) not found in Isaac Lab body names")
    return isaac_representative_body_ids
