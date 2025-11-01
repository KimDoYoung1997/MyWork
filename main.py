"""
Beyond Mimic Sim2Sim MuJoCo Deploy Script

이 스크립트는 논문의 Beyond Mimic 방법론을 구현한 sim-to-sim 배포 시스템입니다.
Isaac Lab에서 학습된 정책을 MuJoCo 환경에서 실행하여 모션 트래킹을 수행합니다.

=== Sim-to-Sim Deploy 핵심 원리 ===

1. 좌표계 독립성 확보:
   - MuJoCo (Z-up) vs Isaac Lab (Y-up) 좌표계 차이에도 불구하고 작동
   - 상대적 관찰값 사용으로 절대 좌표계 차이 흡수
   - 앵커링 메커니즘으로 로봇-모션 데이터 간 상대적 정렬 유지

2. 논문의 Observation 구성 구현:
   o = [c, ξ_{b_anchor}, V_{b_root}, q_joint,r, v_joint,r, a_last]
   - c ∈ ℝ^58 : Reference Motion의 관절 위치 및 속도 (29+29)
   - ξ_{b_anchor} ∈ ℝ^9 : Anchor Body의 자세 추적 오차 (3+6)
   - V_{b_root} ∈ ℝ^6 : Robot's root twist expressed in root frame (3+3)
   - q_joint,r ∈ ℝ^29 : 로봇의 모든 Joint의 현재 각도 (상대값)
   - v_joint,r ∈ ℝ^29 : 로봇의 모든 Joint의 현재 각속도 (절대값)
   - a_last ∈ ℝ^29 : Policy가 직전에 취한 행동 (메모리 역할)

3. Policy Inference 과정:
   - ONNX 모델을 통한 실시간 추론 (50Hz)
   - 앵커링을 통한 좌표계 변환 없이 모션 트래킹
   - PD 제어기를 통한 관절 토크 계산 및 적용

=== 데이터 구조 ===
- NPZ 파일: Isaac Lab에서 export된 모션 데이터
  * body_pos_w: Isaac Lab의 30개 body 순서 (인덱스 9 = torso_link)
  * joint_pos: Reference motion의 관절 위치 (29차원)
  * joint_vel: Reference motion의 관절 속도 (29차원)
- ONNX 모델: Isaac Lab에서 export된 학습된 정책
  * 메타데이터: joint_names, default_joint_pos, action_scale 등

=== 사용법 ===
기본 사용법:
    python main.py

특정 모션과 정책 사용:
    python main.py --motion_file dance1_subject2 --policy_file dance1_subject2

도움말 보기:
    python main.py --help
"""

import argparse
import time
import onnx
from datetime import datetime

import mujoco.viewer
import mujoco
import numpy as np
import torch
from modules.metrics_n_plots import calculate_additional_metrics, save_performance_plots, initialize_tracking_metrics, calculate_and_log_performance_metrics
import onnxruntime
from modules.get_data import get_representative_bodies
from modules.config_loader import get_mujoco_joint_sequence, get_isaac_body_names
from modules.performance_evaluator import generate_final_performance_report
from modules.transforms import compute_relative_transform_mujoco
from modules.controller import pd_control


def parse_arguments():
    """명령행 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="Beyond Mimic Sim2Sim MuJoCo Deploy Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 사용법 (dance1_subject2 사용)
  python main.py



  # 모든 옵션 사용
  python main.py --motion_file dance1_subject2 --policy dance1_subject2_woSE_34000 --duration 100

사용 가능한 모션/정책:
  - dance1_subject2 (기본값)
  - fight1_subject2
        """
    )
    
    parser.add_argument(
        '--motion_file', 
        type=str, 
        default='dance1_subject1',
        help='사용할 모션 파일명 (확장자 제외). 기본값: dance1_subject1'
    )
    
    parser.add_argument(
        '--policy_file', 
        type=str, 
        default='dance1_subject1',
        help='사용할 정책 파일명 (확장자 제외). 기본값: dance1_subject1'
    )
    
    parser.add_argument(
        '--duration', 
        type=float, 
        default=30.0,
        help='시뮬레이션 지속 시간 (초). 기본값: 30.0'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    # 명령행 인자 파싱
    args = parse_arguments()
    
    print("="*60)
    print("Beyond Mimic Sim2Sim MuJoCo Deploy")
    print("="*60)
    print(f"모션 파일: {args.motion_file}")
    print(f"정책 파일: {args.policy_file}")
    print(f"시뮬레이션 시간: {args.duration}초")
    print("="*60)

    # =============================================================================
    # 1. 시뮬레이션 환경 설정
    # =============================================================================
    xml_path = "./unitree_description/mjcf/g1.xml"
    simulation_duration = args.duration                                    # 명령행 인자로 받은 시뮬레이션 시간
    simulation_dt = 0.005                                                   # Isaac Lab과 동일한 시뮬레이션 타임스텝 (0.005초 = 200Hz)
    control_decimation = 4                                                  # Isaac Lab과 동일한 제어기 업데이트 주파수 (simulation_dt * control_decimation = 0.02초; 50Hz)    
    # =============================================================================
    
    # 2. 모션 데이터 로드 (Isaac Lab에서 export된 NPZ 파일)
    # =============================================================================
    motion_file = f"./npzs/{args.motion_file}.npz"
    mocap =  np.load(motion_file)
    mocap_pos = mocap["body_pos_w"]        # 논문의 Reference Motion 위치 데이터 , np.shape(mocap_pos) = (6574, 30, 3)
    mocap_quat = mocap["body_quat_w"]      # 논문의 Reference Motion 자세 데이터 , np.shape(mocap_quat) = (6574, 30, 4)
    mocap_joint_pos = mocap["joint_pos"]   # 논문의 c = [q_joint,m, v_joint,m] 중 관절 위치 부분 , np.shape(mocap_joint_pos) = (6574, 29)
    mocap_joint_vel = mocap["joint_vel"]   # 논문의 c = [q_joint,m, v_joint,m] 중 관절 속도 부분 , np.shape(mocap_joint_vel) = (6574, 29)
    
    # Get motion data length
    motion_length = mocap_joint_pos.shape[0]
    print(f"Motion data length: {motion_length} frames")
    print(f"Maximum simulation time (at 50Hz): {motion_length / 50:.1f} seconds")
    # =============================================================================
    
    # 3. 학습된 정책 로드 (Isaac Lab에서 export된 ONNX 모델)
    # =============================================================================
    policy_path = f"./policies/{args.policy_file}_policy.onnx"
    num_actions = 29    # 29개의 관절 조절 (G1 로봇의 관절 수)
    
    # =============================================================================
    
    # 4. Sim-to-Sim 호환성을 위한 관절 순서 매핑
    # =============================================================================
    # MuJoCo는 XML 파일의 순서대로 관절을 인덱싱하므로, Isaac Lab과 MuJoCo 간의
    # 관절 순서 차이를 해결하기 위한 매핑이 필요함.
    
    # Configuration files에서 관절 순서와 body 이름 로드
    mujoco_joint_seq = get_mujoco_joint_sequence()
    isaac_body_names = get_isaac_body_names()    

    # =============================================================================
    
    # 5. ONNX 모델 메타데이터 파싱 (Sim-to-Sim 호환성 확보)
    # =============================================================================
    # Isaac Lab에서 export된 ONNX 모델의 메타데이터를 읽어서 MuJoCo 환경의 설정을 동기화합니다.
    # 이를 통해 sim2sim 변환을 달성하고 정책이 올바르게 동작하도록 합니다.
    rl_model: onnx.ModelProto = onnx.load(policy_path)

    # Observation structure tracking
    observation_names = []
    has_motion_anchor_pos_b = False
    has_motion_anchor_ori_b = False
    has_base_lin_vel = False
    has_base_ang_vel = False

    # Isaac Lab에서 RL 정책 훈련을 isaac_joint_seq 순서로 학습했으므로,
    # MuJoCo에서 실행할 때는 mujoco_joint_seq(g1.xml) 순서로 변환해야 합니다.
    for prop in rl_model.metadata_props:
        if prop.key == "joint_names":
            # Isaac Lab에서 학습된 정책이 사용하는 관절 순서 (29개)
            # 논문의 q_joint,r, v_joint,r 계산 시 이 순서를 따라야 합니다.
            isaac_joint_seq: list[str] = prop.value.split(",")
            
        if prop.key == "default_joint_pos":  
            # Isaac Lab에서 사용한 초기 기본 관절 위치 (중립 자세)
            # 논문의 q_joint,r 계산 시 상대값을 구하기 위해 사용됩니다.
            isaac_joint_pos_array = np.array([float(x) for x in prop.value.split(",")])
            # MuJoCo 순서로 변환 (Sim-to-Sim 호환성)하여 MuJoCo상의 default_joint_pos 배열 생성
            mujoco_initial_target_joint_pos = np.array([isaac_joint_pos_array[isaac_joint_seq.index(joint)] for joint in mujoco_joint_seq])
            
        if prop.key == "joint_stiffness":
            # PD 제어기에서 사용할 관절 강성 계수
            isaac_stiffness_array = np.array([float(x) for x in prop.value.split(",")])
            mujoco_stiffness_array = np.array([isaac_stiffness_array[isaac_joint_seq.index(joint)] for joint in mujoco_joint_seq])
            
        if prop.key == "joint_damping":
            # PD 제어기에서 사용할 관절 감쇠 계수
            isaac_damping_array = np.array([float(x) for x in prop.value.split(",")])
            mujoco_damping_array = np.array([isaac_damping_array[isaac_joint_seq.index(joint)] for joint in mujoco_joint_seq])
        
        if prop.key == "action_scale":
            # Policy 출력을 실제 joint 위치로 변환하는 스케일 팩터.
            # 이는 Mujoco로 변환을 하지 않는데, 
            # 논문의 액션 스케일링에 해당합니다.
            isaac_action_scale_array = np.array([float(x) for x in prop.value.split(",")])
        
        if prop.key == "observation_names":
            # Parse observation structure
            observation_names = prop.value.split(",")
            has_motion_anchor_pos_b = "motion_anchor_pos_b" in observation_names
            has_motion_anchor_ori_b = "motion_anchor_ori_b" in observation_names
            has_base_lin_vel = "base_lin_vel" in observation_names
            has_base_ang_vel = "base_ang_vel" in observation_names
            
        print(f"{prop.key}: {prop.value}")
        print("\n")
    
    # Calculate observation dimension dynamically
    # motion_anchor_pos_b: 3D position (3)
    # motion_anchor_ori_b: rotation matrix (6)
    num_obs = 0
    has_both_anchor = has_motion_anchor_pos_b and has_motion_anchor_ori_b
    
    for obs_name in observation_names:
        if obs_name == "command":
            num_obs += 58  # reference motion (29 + 29)
        elif obs_name == "motion_anchor_pos_b": # woSE 일 경우 None
            num_obs += 3  # 3D position
        elif obs_name == "motion_anchor_ori_b":
            # motion_anchor_ori_b is always 6 dimensions (rotation matrix 2x3)
            num_obs += 6
        elif obs_name == "base_lin_vel":
            num_obs += 3  # 3D linear velocity
        elif obs_name == "base_ang_vel":
            num_obs += 3  # 3D angular velocity
        elif obs_name == "joint_pos":
            num_obs += num_actions  # 29 joint positions
        elif obs_name == "joint_vel":
            num_obs += num_actions  # 29 joint velocities
        elif obs_name == "actions":
            num_obs += num_actions  # 29 previous actions
    
    print(f"Observation dimension: {num_obs}")
    print(f"Observation structure: {observation_names}")
    print("\n")    # =============================================================================


    # =============================================================================

    # 6. MuJoCo 물리 시뮬레이션 환경 로드
    mj_model = mujoco.MjModel.from_xml_path(xml_path)      # 물리 시뮬레이션 환경 정의
    mj_data = mujoco.MjData(mj_model)                     # 물리 시뮬레이션 상태 관리
    mj_model.opt.timestep = simulation_dt                 # Isaac Lab과 동일한 타임스텝 설정 : 200Hz

    # Isaac Lab에서 export된 ONNX 정책 로드
    policy = onnxruntime.InferenceSession(policy_path)
    # ONNX 정책 입력/출력 이름 (사용되지 않지만 참고용으로 유지)

    # observation중 하나로, 이전 action의 버퍼 (논문의 a_last)
    a_last: np.ndarray = np.zeros((num_actions,), dtype=np.float32)  

    anchor_body_name = "torso_link"
    # 초기 모션 데이터 (실제로는 루프 내에서 업데이트됨)
    mujoco_current_target_pos = mujoco_initial_target_joint_pos.copy()             # 시뮬레이터가 (시작했을때 초기 관절 위치 배열을 mujoco_joint_pos_array에 저장

    # g1.xml에서 position, orientation 초기화 이후 (IsaacLab -> MuJoCo) joint 위치(자세) 초기화
    mj_data.qpos[7:] = mujoco_current_target_pos                         # anchor body(torso)에 한해서는  $\hat T_{b_{anchor,r}}$ 와  $T_{b_{anchor,m}}$ 이 개념적으로 같다      

    # =============================================================================

    # 트래킹 성능 로깅을 위한 변수들 초기화
    mujoco_anchor_body_id, representative_body_ids, isaac_representative_body_ids, additional_metrics = initialize_tracking_metrics(
        mj_model, anchor_body_name, isaac_body_names
    )
    
    # 대표 body들 정의 (함수 호출에서 사용)
    representative_bodies = get_representative_bodies()

    # =============================================================================

    timestep = 0
    obs: np.ndarray = np.zeros(num_obs, dtype=np.float32)        # 논문의 o = [c, ξ_{b_anchor}, V_{b_root}, q_joint,r, v_joint,r, a_last] (160차원)
    isaac_anchor_body_id: int = isaac_body_names.index(anchor_body_name)  # Isaac Lab에서는 9
    counter = 0 # 제어 신호 적용 횟수
    log_interval = 100  # 100 스텝마다 로깅

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            # =============================================================================
            # 7.1 물리 시뮬레이션 스텝 실행 (200Hz)
            # =============================================================================
            mujoco.mj_step(mj_model, mj_data)  # MuJoCo 물리 시뮬레이션 진행
            
            # =============================================================================
            # 7.2 PD 제어기를 통한 관절 토크 계산 및 적용
            # =============================================================================
            # 정책에서 출력된 목표 관절 위치를 PD 제어기로 토크 변환, np.shape(tau) = (29,)
            tau: np.ndarray = pd_control(
                target_q=mujoco_current_target_pos,           # 정책이 출력한 (Isaac -> Mujoco)목표 관절 위치
                current_q=mj_data.qpos[7:],        # 현재 관절 위치
                kp=mujoco_stiffness_array,                # 관절 강성 계수
                target_dq=np.zeros_like(mujoco_damping_array),  # 목표 관절 속도 (0으로 설정)
                current_dq=mj_data.qvel[6:],       # 현재 관절 속도
                kd=mujoco_damping_array                   # 관절 감쇠 계수
            )
            mj_data.ctrl[:] : np.ndarray = tau  # 계산된 토크를 액추에이터에 적용
            
            counter += 1
            # =============================================================================
            # 7.3 정책 추론 및 관찰값 계산 (50Hz - control_decimation=4)
            # =============================================================================
            if counter % control_decimation == 0:
                # =============================================================================
                # 7.3.1 현재 로봇 상태 및 목표 모션 데이터 추출
                # =============================================================================
                # Clamp timestep to motion data range to prevent index out of bounds
                safe_timestep = min(timestep, motion_length - 1)
                
                # mujoco_anchor_body_id = 16
                mujoco_robot_anchor_pos: np.ndarray = mj_data.xpos[mujoco_anchor_body_id]              # 현재 로봇 앵커 바디 위치 (torso_link) , eg) array([-3.9635000e-03, -3.5901179e-21,  8.3332125e-01])
                mujoco_robot_anchor_quat: np.ndarray = mj_data.xquat[mujoco_anchor_body_id]           # 현재 로봇 앵커 바디 자세 (torso_link)
                
                # 논문의 c = [q_joint,m, v_joint,m] 구성 (Reference phase) : 이 vector는 매 time step마다 policy에 입력되어 로봇의 다음 행동을 결정하는 데 사용됩니다.
                mocap_reference_phase = np.concatenate((mocap_joint_pos[safe_timestep,:],mocap_joint_vel[safe_timestep,:]),axis=0)    # shape : (58,) = (29+29)의 concat
                
                # 목표 모션의 앵커 바디 상태, c ∈ ℝ^58
                mocap_anchor_pos = mocap_pos[safe_timestep, isaac_anchor_body_id, :]  # 목표 모션 앵커 바디 위치 eg) array([-3.6416985e-03, -7.5313076e-04,  8.4310246e-01], dtype=float32)
                mocap_anchor_quat = mocap_quat[safe_timestep, isaac_anchor_body_id, :]  # 목표 모션 앵커 바디 자세
                
                # =============================================================================
                # 7.3.2 앵커링을 통한 상대 변환 계산 (논문의 ξ_{b_anchor})
                # =============================================================================
                # Sim-to-Sim 핵심: 좌표계 변환 없이 상대적 관계 계산
                # anchor_pos_track_erro : 논문의 ξ_{b_anchor} 위치 부분
                # anchor_quat_track_error : 논문의 ξ_{b_anchor} 회전 부분
                # mujoco 좌표계에서 계산된 anchor_pos_track_error, anchor_quat_track_error 는 모션 기준에서 로봇 기준으로 변환된 값이다.
                anchor_pos_track_error, temp_anchor_quat_track_error = compute_relative_transform_mujoco(
                    mujoco_robot_anchor_pos_A=mujoco_robot_anchor_pos,    # 로봇 기준
                    mujoco_robot_anchor_quat_A=mujoco_robot_anchor_quat,  # 로봇 기준
                    isaac_ref_pos_B=mocap_anchor_pos,    # 모션 기준
                    isaac_ref_quat_B=mocap_anchor_quat   # 모션 기준
                ) # timestep = 0 일때  anchor_pos_track_error = 0 0 0 에 가깝고  anchor_quat_track_error = 1 0 0 0 에 가깝다.
                
                # 회전 행렬을 6차원 벡터로 변환 (논문의 ξ_{b_anchor} 회전 부분)
                anchor_quat_track_error = np.zeros(9)
                mujoco.mju_quat2Mat(anchor_quat_track_error, temp_anchor_quat_track_error)    # convert quaternion to 3D rotation matrix, 초기화된 anchor_ori 에 anchor_quat_track_error(quaternion) 의 회전 행렬 저장 , anchor_ori.shape=(9,)
                anchor_quat_track_error = anchor_quat_track_error.reshape(3, 3)[:, :2]  # 첫 2열만 사용 (6차원)
                anchor_quat_track_error = anchor_quat_track_error.reshape(-1,)
                
                # =============================================================================
                # 7.3.3 논문의 Observation 구성 구현 (Dynamic based on policy)
                # =============================================================================
                # Build observation dynamically based on policy's observation_names
                offset = 0
                
                for obs_name in observation_names:
                    if obs_name == "command":
                        # Reference motion: joint position (29) + joint velocity (29) = 58
                        obs[offset:offset + 58] = mocap_reference_phase
                        offset += 58
                        
                    elif obs_name == "motion_anchor_pos_b":
                        # Anchor body position error: 3D position
                        obs[offset:offset + 3] = anchor_pos_track_error
                        offset += 3
                        
                    elif obs_name == "motion_anchor_ori_b":
                        # Anchor body orientation error: rotation matrix (6)
                        obs[offset:offset + 6] = anchor_quat_track_error
                        offset += 6
                            
                    elif obs_name == "base_lin_vel":
                        # Robot's linear velocity in root frame
                        obs[offset:offset + 3] = mj_data.qvel[0:3]
                        offset += 3
                        
                    elif obs_name == "base_ang_vel":
                        # Robot's angular velocity in root frame
                        obs[offset:offset + 3] = mj_data.qvel[3:6]
                        offset += 3
                        
                    elif obs_name == "joint_pos":
                        # Robot's joint positions (relative to default)
                        qpos_xml = mj_data.qpos[7 : 7 + num_actions]
                        qpos_seq = np.array([qpos_xml[mujoco_joint_seq.index(joint)] for joint in isaac_joint_seq])
                        obs[offset:offset + num_actions] = qpos_seq - isaac_joint_pos_array
                        offset += num_actions
                        
                    elif obs_name == "joint_vel":
                        # Robot's joint velocities
                        qvel_xml = mj_data.qvel[6 : 6 + num_actions]
                        qvel_seq = np.array([qvel_xml[mujoco_joint_seq.index(joint)] for joint in isaac_joint_seq])
                        obs[offset:offset + num_actions] = qvel_seq
                        offset += num_actions
                        
                    elif obs_name == "actions":
                        # Previous actions (policy memory)
                        obs[offset:offset + num_actions] = a_last
                        offset += num_actions

                # =============================================================================
                # 7.3.4 ONNX 정책 추론 실행
                # =============================================================================
                # Isaac Lab에서 학습된 정책을 MuJoCo에서 실행
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)  # 배치 차원 추가
                action = policy.run(['actions'], {
                    'obs': obs_tensor.numpy(),
                    'time_step': np.array([timestep], dtype=np.float32).reshape(1,1)
                })[0]
                action = np.asarray(action).reshape(-1)  # 정책 출력 (29차원)
                a_last = action.copy()  # 다음 스텝을 위한 메모리 저장
                
                # =============================================================================
                # 7.3.5 정책 출력을 실제 관절 위치로 변환
                # =============================================================================
                # 논문의 액션 스케일링: q_{j,t} = α_j * a_{j,t} + q̄_j
                # α_j: action_scale, a_{j,t}: 정책 출력, q̄_j: 기본 관절 위치
                isaac_current_target_pos = action * isaac_action_scale_array + isaac_joint_pos_array    # policy는 isaac_joint_pos_array 기준으로 학습됨
                isaac_current_target_pos = isaac_current_target_pos.reshape(-1,)                        
                # Isaac Lab 순서에서 MuJoCo 순서로 변환 (Sim-to-Sim 호환성)
                # mujoco_current_target_pos 는 다시 pd_control 제어기에 입력되어 관절 토크 계산에 사용됨
                mujoco_current_target_pos = np.array([isaac_current_target_pos[isaac_joint_seq.index(joint)] for joint in mujoco_joint_seq]) # isaac_joint_seq 순서를 mujoco_joint_seq 순서로 변환
                
                # mujoco_initial_target_joint_pos =np.array([isaac_joint_pos_array[isaac_joint_seq.index(joint)] for joint in mujoco_joint_seq])

                calculate_and_log_performance_metrics(
                    print_log=True,
                    timestep=timestep,
                    mj_data=mj_data,
                    mocap_joint_pos=mocap_joint_pos,
                    mocap_joint_vel=mocap_joint_vel,
                    mocap_pos=mocap_pos,
                    mocap_quat=mocap_quat,
                    robot_anchor_pos=mujoco_robot_anchor_pos,
                    robot_anchor_quat=mujoco_robot_anchor_quat,
                    mocap_anchor_pos=mocap_anchor_pos,
                    mocap_anchor_quat=mocap_anchor_quat,
                    additional_metrics=additional_metrics,
                    log_interval=log_interval,
                    mujoco_joint_seq=mujoco_joint_seq,
                    isaac_joint_seq=isaac_joint_seq,
                    representative_body_ids=representative_body_ids,
                    isaac_representative_body_ids=isaac_representative_body_ids,
                    representative_bodies=representative_bodies,
                    motion_length=motion_length
                )
                    
                # print(f"mj_data.qpos: {mj_data.qpos}\n")
                # print(f"mj_data.qvel: {mj_data.qvel}\n")
                # print(f"normalized action from policy: {action}\n")
                # print(f"mujoco_current_target_pos: {mujoco_current_target_pos}\n")
                            
                timestep+=1
                

            # =============================================================================
            # 7.4 시뮬레이션 동기화 및 시간 관리
            # =============================================================================
            viewer.sync()   # MuJoCo 뷰어와 시뮬레이션 데이터 동기화
            
            # Isaac Lab과 동일한 시뮬레이션 속도 유지
            time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)   # 200Hz 유지를 위한 대기 시간 계산
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)



    # =============================================================================
    # 8. Sim-to-Sim Deploy 성능 요약 및 분석 (commands.py 기반)
    # =============================================================================
    print("\n" + "="*60)
    print("Sim-to-Sim Deploy 완료 - Beyond Mimic 성능 요약 (commands.py 기반)")
    print("="*60)

    if additional_metrics['error_anchor_body_pos']:
        # =============================================================================
        # 최종 성능 평가 및 결과 리포트
        # =============================================================================
        performance_metrics, success_level = generate_final_performance_report(
            additional_metrics, simulation_dt, control_decimation
        )
    else:
        print("❌ 경고: 성능 데이터가 기록되지 않았습니다.")
        print("   시뮬레이션이 정상적으로 실행되지 않았을 수 있습니다.")
        
        

    # =============================================================================
    # 9. 성능 플롯 생성 및 저장
    # =============================================================================
    print("\n" + "="*60)
    print("성능 플롯 생성 중...")
    print("="*60)

    # Extract policy suffix from policy_file by removing motion_file prefix
    # Example: fight1_subject2_woSE_5500 -> woSE_5500
    policy_suffix = None
    if args.policy_file.startswith(args.motion_file):
        suffix_with_underscore = args.policy_file[len(args.motion_file):]
        if suffix_with_underscore.startswith("_"):
            policy_suffix = suffix_with_underscore[1:]  # Remove leading underscore
        else:
            policy_suffix = suffix_with_underscore
    
    # If policy_suffix couldn't be extracted, use the full policy_file name
    if not policy_suffix:
        policy_suffix = args.policy_file

    # 성능 플롯 저장 (commands.py 기반 지표만)
    save_performance_plots(
        additional_metrics, 
        save_dir=f"./performance_plots/{args.motion_file}_motion",
        policy_suffix=policy_suffix,
        motion_file=args.motion_file,
        policy_file=args.policy_file
    )

    print("="*60)
    print("Beyond Mimic Sim-to-Sim Deploy 완료")
    print("="*60)
