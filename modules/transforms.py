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

def transform_velocity_to_local_frame(world_vel, body_quat):
    """MuJoCo의 global frame linear velocity를 local frame으로 변환
    
    === 왜 이 함수가 필요한가? ===
    
    MuJoCo 제공:  qvel[0:3] = Linear velocity in GLOBAL (world) frame
                  출처: https://github.com/google-deepmind/mujoco/issues/691
    
    논문 요구사항: V_b_root = Robot's root twist expressed in LOCAL (root) frame
                  "expressed in root frame" - BeyondMimic paper
    
    → GLOBAL → LOCAL 변환 필수!
    
    Args:
        world_vel: GLOBAL (world) frame에서의 속도 벡터 (3,) [vx, vy, vz]
                   MuJoCo qvel[0:3]에서 가져온 값
        body_quat: Body의 orientation quaternion (4,) [w, x, y, z]
                   MuJoCo xquat에서 가져온 값
        
    Returns:
        local_vel: LOCAL (body/root) frame에서의 속도 벡터 (3,) [vx', vy', vz']
                   Policy에 제공할 값
        
    === 수학적 배경 ===
    v_local = R^T * v_global
    - v_global: GLOBAL (world) frame에서 표현된 속도 벡터
    - R: World → Body 회전 행렬 (body_quat로부터)
    - R^T: 회전 행렬의 전치 (역회전)
    - v_local: LOCAL (body) frame에서 표현된 속도 벡터
    
    [참고: Angular velocity는?]
    MuJoCo qvel[3:6]는 이미 LOCAL frame이므로 변환 불필요!
    출처: https://github.com/google-deepmind/mujoco/issues/691
    """
    # Step 1: Convert quaternion to rotation matrix
    # R represents the orientation of the body in the GLOBAL frame
    R = quat_to_rotation_matrix(body_quat)  # 3x3 rotation matrix
    
    # Step 2: Transform velocity from GLOBAL to LOCAL frame
    # v_local = R^T * v_global (inverse rotation)
    # R^T transforms vectors from GLOBAL frame to LOCAL (body) frame
    local_vel = R.T @ world_vel
    
    return local_vel


def compute_relative_transform_mujoco(mujoco_robot_anchor_pos_A, mujoco_robot_anchor_quat_A, isaac_ref_pos_B, isaac_ref_quat_B):
    """로봇 body frame 기준 상대 변환 계산 - 논문의 ξ_{b_anchor} 계산에 사용
    
    "로봇 앵커(A)를 기준으로 reference 모션 앵커(B)가 어디에/어떻게 위치하는가?"를 계산합니다.
    
    Args:
        mujoco_robot_anchor_pos_A: 현재 로봇 앵커 바디 position (3,) [x, y, z], world frame
        mujoco_robot_anchor_quat_A: 현재 로봇 앵커 바디 orientation (4,) [w, x, y, z]
        isaac_ref_pos_B: reference 모션 앵커 바디 position (3,) [x, y, z], world frame
        isaac_ref_quat_B: reference 모션 앵커 바디 orientation (4,) [w, x, y, z]
            
    Returns:
        rel_pos: 로봇 body frame 기준 상대 위치 (3,) - 논문의 ξ_{b_anchor} 위치 부분
        rel_quat: 로봇 body frame 기준 상대 회전 (4,) [w,x,y,z] - 논문의 ξ_{b_anchor} 회전 부분
        
    === 수학적 배경 ===
    - T_A: Robot anchor의 world frame transformation matrix (4x4)
    - T_B: Mocap anchor의 world frame transformation matrix (4x4)
    - T_rel = T_A^(-1) * T_B: Robot body frame 기준에서 Mocap의 상대 변환
    
    === 논문과의 연관성 ===
    논문의 Observation 구성: o = [c, **ξ_{b_anchor}**, V_{b_root}, q_joint,r, v_joint,r, a_last]
    이 함수의 출력은 **ξ_{b_anchor}** ∈ ℝ^9 (3+6) 계산에 사용됩니다:
    - rel_pos (3,) → anchor_pos_track_error (position part)
    - rel_quat (4,) → anchor_quat_track_error (6D rotation representation으로 변환 후 사용)
    
    === Sim-to-Sim 핵심 ===
    이 함수는 robot body frame 기준 상대 변환을 계산하므로, MuJoCo와 Isaac Lab의
    절대 좌표계(Z-up vs Y-up) 차이가 있어도 상대적 관계는 동일하게 계산됩니다.
    """
    # 1. 4x4 transformation matrices 생성
    T_A = pose_to_transformation_matrix(mujoco_robot_anchor_pos_A, mujoco_robot_anchor_quat_A)  # Robot anchor의 world frame pose
    T_B = pose_to_transformation_matrix(isaac_ref_pos_B, isaac_ref_quat_B)  # Mocap anchor의 world frame pose
    
    # 2. 상대 변환 계산: T_rel = T_A^(-1) * T_B
    # Robot body frame 기준에서 Mocap의 위치/자세를 표현
    T_A_inv = np.linalg.inv(T_A)  # Robot frame의 역변환 (world → robot body frame)
    T_rel = T_A_inv @ T_B         # 상대 변환 행렬 (robot body frame 기준)
    
    # 3. 결과 추출
    rel_pos = T_rel[0:3, 3]        # 상대 위치 (3,) robot body frame에서 표현된 mocap anchor 위치
    rel_rotation = T_rel[0:3, 0:3] # 상대 회전 (3x3) robot body frame에서 표현된 mocap anchor 자세
    
    # 4. 회전 행렬을 쿼터니언으로 변환
    rel_quat: np.ndarray = rotation_matrix_to_quaternion(rel_rotation)  # (4,) [w,x,y,z]
    
    return rel_pos, rel_quat
