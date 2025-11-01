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
    
    === 이 함수를 사용하지 않으면? ===
    
    예시 상황: 로봇이 90도 왼쪽으로 회전한 상태
    
    [변환 전 - 잘못된 방법]
    Robot state:
      - GLOBAL frame에서 북쪽(+Y)을 향해 이동 중
      - 로봇은 동쪽(+X)을 향함 (90도 회전)
      
    global_vel = [0.0, 1.0, 0.0]  # MuJoCo qvel[0:3]: GLOBAL frame에서 북쪽으로 1 m/s
    
    잘못된 코드:
      obs[...] = global_vel  #  GLOBAL frame 그대로 사용 (틀림!)
      
    Policy가 받는 정보:
      "로봇의 앞쪽(+X local)으로 0 m/s, 왼쪽(+Y local)으로 1 m/s"
      → 하지만 실제로는 로봇 기준 왼쪽으로 이동 중!
      → Policy가 혼란스러워함 (좌표계 불일치!)
    
    [변환 후 - 올바른 방법]
    body_quat = [0.707, 0, 0, 0.707]  # MuJoCo xquat: 90도 Z축 회전
    local_vel = transform_velocity_to_local_frame(global_vel, body_quat)
    # local_vel = [1.0, 0.0, 0.0]   LOCAL frame에서 전방으로 1 m/s
    
    올바른 코드:
      obs[...] = local_vel  #  LOCAL frame 변환 후 사용 (맞음!)
      
    Policy가 받는 정보:
      "로봇의 앞쪽(+X local)으로 1 m/s, 왼쪽(+Y local)으로 0 m/s"
      → 올바름! 로봇이 자신의 전방으로 이동하고 있다고 정확히 인식
      → Policy가 올바른 제어 명령 생성 가능
    
    [실제 성능 차이]
    - 변환 없이 사용 (GLOBAL): Policy 발산, 로봇 넘어짐 (Sim-to-Sim gap)
    - 변환 후 사용 (LOCAL): Policy 안정적, 성능 "무지하게 좋아짐"
    
    [핵심 교훈]
    같은 속도 벡터라도 좌표계에 따라 의미가 완전히 다름!
    
    MuJoCo 제공:     GLOBAL [0, 1, 0] = "세계 기준 북쪽으로 1 m/s"
    논문 요구:       LOCAL [0, 1, 0] = "로봇 기준 왼쪽으로 1 m/s"
                     ↑ 완전히 다른 의미!
    
    → Policy는 LOCAL frame 속도를 기대하므로 변환 필수!
    → MuJoCo qvel[0:3] (GLOBAL) → transform → LOCAL → Policy
    
    [참고: Angular velocity는?]
    MuJoCo qvel[3:6]는 이미 LOCAL frame이므로 변환 불필요!
    출처: https://github.com/google-deepmind/mujoco/issues/691
    """
    local_vel = np.zeros(3)
    quat_inv = np.zeros(4)
    
    # Step 1: Quaternion conjugate (inverse rotation for unit quaternions)
    # Compute q^(-1) = q* (conjugate) for unit quaternion
    mujoco.mju_negQuat(quat_inv, body_quat)
    
    # Step 2: Rotate velocity vector from GLOBAL frame to LOCAL frame
    # v_local = R^T * v_global = rotate(v_global, q^(-1))
    mujoco.mju_rotVecQuat(local_vel, world_vel, quat_inv)
    
    return local_vel


def compute_relative_transform_mujoco(mujoco_robot_anchor_pos_A, mujoco_robot_anchor_quat_A, isaac_ref_pos_B, isaac_ref_quat_B):
    """
    이 함수는 논문의 ξ_{b_anchor} 계산하기 위해 MuJoCo vs Isaac Lab 좌표계 차이를 보정함
    "로봇(mujoco_robot_anchor_pos_A)을 기준으로 reference 모션(isaac_ref_pos_B)이 어디에/어떻게 위치하는가?"
    
    Args:
        mujoco_robot_anchor_pos_A : 현재 Mujoco 로봇 앵커 바디 position (3,) [x, y, z], MuJoCo좌표계 기준 
        mujoco_robot_anchor_quat_A : 현재 Mujoco 로봇 앵커 바디 orientation (4,) [w, x, y, z], MuJoCo좌표계 기준 
        isaac_ref_pos_B : reference 모션 (Isaac) 앵커 바디 position (3,) [x, y, z], Isaac좌표계 기준 
        isaac_ref_quat_B : reference 모션 (Isaac) 앵커 바디 orientation (4,) [w, x, y, z], Isaac좌표계 기준 
            
    Returns:
        rel_pos: 로봇 기준 reference 모션의 상대 위치 (3,) - 논문의 ξ_{b_anchor} 위치 부분
        rel_quat: 로봇 기준 reference 모션의 상대 회전 (4,) - 논문의 ξ_{b_anchor} 회전 부분
        
    === 수학적 배경 ===
    - T_A: Robot frame (mujoco_robot_anchor_pos_A)의 transformation matrix (현재 Mujoco 로봇 상태)
    - T_B: Mocap frame (isaac_ref_pos_B)의 transformation matrix (목표 Reference Isaac 모션 상태)
    - T_rel = T_A^(-1) * T_B: Robot 기준에서 Mocap의 상대 변환 (A --> B == B-A)
    
    
    === 논문과의 연관성 ===
    논문의 Observation 구성: o = [c, **ξ_{b_anchor}**, V_{b_root}, q_joint,r, v_joint,r, a_last]
    이 함수는 **ξ_{b_anchor}** ∈ ℝ^9 (3+6) : anchor_pos_track_error(3) + anchor_quat_track_error(6) 계산에 사용됩니다.
    anchor_pos_track_error = reference 모션의 anchor position과 현재 로봇의 anchor position 간의 오차
    anchor_quat_track_error = reference 모션의 anchor quaternion과 현재 로봇의 anchor quaternion 간의 오차
    
    === 중요 ===
    이 함수는 이미 좌표계 변환을 포함하고 있습니다!
    - T_A^(-1): Robot body frame으로의 좌표계 변환
    - T_rel의 position 부분: 이미 robot body frame에서 표현된 상대 위치
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
