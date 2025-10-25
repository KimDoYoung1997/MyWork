"""
Transform utilities for robot motion tracking
This module provides functions for coordinate transformations, quaternion operations, and relative pose calculations
"""

import numpy as np
import mujoco

def quat_to_rotation_matrix(quat):
    """쿼터니언을 3x3 회전 행렬로 변환
    
    Args:
        quat: 쿼터니언 (4,) [w, x, y, z]
        
    Returns:
        3x3 회전 행렬
    """
    rotm = np.zeros(9)
    mujoco.mju_quat2Mat(rotm, quat)
    return rotm.reshape(3, 3)

def pose_to_transformation_matrix(pos, quat):
    """위치와 쿼터니언을 4x4 transformation matrix로 변환
    
    Args:
        pos: 위치 벡터 (3,) [x, y, z]
        quat: 쿼터니언 (4,) [w, x, y, z]
        
    Returns:
        T: 4x4 transformation matrix
           [[R11, R12, R13, tx],
            [R21, R22, R23, ty],
            [R31, R32, R33, tz],
            [ 0,   0,   0,  1]]
    """
    T = np.eye(4)
    T[0:3, 0:3] = quat_to_rotation_matrix(quat)  # 회전 부분
    T[0:3, 3] = pos                              # 평행이동 부분
    return T

def rotation_matrix_to_quaternion(R):
    """3x3 회전 행렬을 쿼터니언으로 변환 [w, x, y, z]
    
    Args:
        R: 3x3 회전 행렬
        
    Returns:
        quat: 정규화된 쿼터니언 (4,) [w, x, y, z]
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    
    quat = np.array([qw, qx, qy, qz])
    return quat / np.linalg.norm(quat)  # 정규화

def compute_relative_transform_mujoco(mujoco_robot_anchor_pos_A, mujoco_robot_anchor_quat_A, isaac_ref_pos_B, isaac_ref_quat_B):
    """
    Sim-to-Sim Deploy 핵심 함수: 앵커링을 통한 상대 변환 계산
    
    이 함수는 논문의 ξ_{b_anchor} 계산
    좌표계 변환 없이도 모션 트래킹이 가능하도록 합니다.
    
    mujoco_robot_anchor_pos_A : 현재 Mujoco 로봇 앵커 바디 position, MuJoCo좌표계 기준 ($F_{\text{world,r=mujoco}}$ )
    mujoco_robot_anchor_quat_A : 현재 Mujoco 로봇 앵커 바디 orientation, MuJoCo좌표계 기준 ($F_{\text{world,r=mujoco}}$ )
    isaac_ref_pos_B : 목표 모션 (Isaac) 앵커 바디 position, Isaac좌표계 기준 ($F_{\text{world,m=isaac}}$ )
    isaac_ref_quat_B : 목표 모션 (Isaac) 앵커 바디 orientation, Isaac좌표계 기준 ($F_{\text{world,m=isaac}}$ )
    
    === 수학적 배경 ===
    - T_A: Robot frame의 transformation matrix (현재 Mujoco 로봇 상태)
    - T_B: Mocap frame의 transformation matrix (목표 Reference Isaac 모션 상태)
    - T_rel = T_A^(-1) * T_B: Robot 기준에서 Mocap의 상대 변환 (A->B == B-A)
    
    === 물리적 의미 ===
    "로봇을 기준으로 목표 모션이 어디에/어떻게 위치하는가?"
    이 상대 변환을 통해 좌표계 차이를 흡수하고 모션 트래킹을 수행합니다.
    
    === 논문과의 연관성 ===
    논문의 Observation 구성: o = [c, **ξ_{b_anchor}**, V_{b_root}, q_joint,r, v_joint,r, a_last]
    이 함수는 **ξ_{b_anchor}** ∈ ℝ^9 (3+6) : anchor_pos_track_error(3) + anchor_quat_track_error(6) 계산에 사용됩니다.
    anchor_pos_track_error = Reference motion의 anchor position과 현재 로봇의 anchor position 간의 오차
    anchor_quat_track_error = Reference motion의 anchor quaternion과 현재 로봇의 anchor quaternion 간의 오차
    
    === Sim-to-Sim에서의 역할 ===
    1. 좌표계 독립성: MuJoCo vs Isaac Lab 좌표계 차이 흡수
    2. 앵커링: 로봇과 모션 데이터 간의 상대적 정렬 유지
    3. 정규화: 절대 좌표계 대신 상대적 관계에 집중
    
    Args:
        mujoco_robot_anchor_pos_A: 로봇 앵커 바디의 현재 위치 (3,) [x, y, z]
        mujoco_robot_anchor_quat_A: 로봇 앵커 바디의 현재 자세 (4,) [w, x, y, z]
        isaac_ref_pos_B: 모션 데이터의 앵커 바디 위치 (3,) [x, y, z]  
        isaac_ref_quat_B: 모션 데이터의 앵커 바디 자세 (4,) [w, x, y, z]
        
    Returns:
        rel_pos: 로봇 기준 모션의 상대 위치 (3,) - 논문의 ξ_{b_anchor} 위치 부분
        rel_quat: 로봇 기준 모션의 상대 회전 (4,) - 논문의 ξ_{b_anchor} 회전 부분
    """
    # 1. 4x4 transformation matrices 생성
    T_A = pose_to_transformation_matrix(mujoco_robot_anchor_pos_A, mujoco_robot_anchor_quat_A)  # Robot frame
    T_B = pose_to_transformation_matrix(isaac_ref_pos_B, isaac_ref_quat_B)  # Mocap frame
    
    # 2. 상대 변환 계산: T_rel = T_A^(-1) * T_B
    T_A_inv = np.linalg.inv(T_A)  # Robot frame의 역변환
    T_rel = T_A_inv @ T_B         # 상대 변환 행렬
    
    # 3. 결과 추출
    rel_pos = T_rel[0:3, 3]        # 상대 위치 (A->B == B-A) , A : Mujoco, B : Isaac
    rel_rotation = T_rel[0:3, 0:3] # 상대 회전 (A->B == B-A) , A : Mujoco, B : Isaac
    
    # 4. 회전 행렬을 쿼터니언으로 변환
    rel_quat: np.ndarray = rotation_matrix_to_quaternion(rel_rotation)  # 1 0 0 0 
    
    return rel_pos, rel_quat
