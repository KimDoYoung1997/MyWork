import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import mujoco
from modules.get_data import get_representative_bodies, get_mujoco_body_ids, get_isaac_body_indices

def quat_error_magnitude(q1, q2):
    """쿼터니언 간의 각도 차이를 계산합니다 (Isaac Lab의 quat_error_magnitude와 동일)"""
    # 쿼터니언 내적을 통한 각도 차이 계산
    dot_product = np.abs(np.dot(q1, q2))
    dot_product = np.clip(dot_product, 0.0, 1.0)  # 수치 안정성 확보
    return 2 * np.arccos(dot_product)



def calculate_additional_metrics(robot_anchor_pos_w, robot_anchor_quat, mocap_anchor_pos_w, mocap_anchor_quat,
                                robot_joint_pos, mocap_joint_pos, robot_joint_vel, mocap_joint_vel,
                                robot_body_pos, mocap_body_pos, robot_body_quat, mocap_body_quat,
                                robot_body_lin_vel_w, mocap_body_lin_vel_w, robot_body_ang_vel_w, mocap_body_ang_vel_w):
    """
    commands.py 기반 추가 성능 지표를 계산합니다.
    
    Args:
        robot_anchor_pos: 로봇 앵커 위치 (3,)
        robot_anchor_quat: 로봇 앵커 쿼터니언 (4,)
        mocap_anchor_pos: 목표 앵커 위치 (3,)
        mocap_anchor_quat: 목표 앵커 쿼터니언 (4,)
        robot_joint_pos: 로봇 관절 위치 (29,)
        mocap_joint_pos: 목표 관절 위치 (29,)
        robot_joint_vel: 로봇 관절 속도 (29,)
        mocap_joint_vel: 목표 관절 속도 (29,)
        robot_body_pos: 로봇 바디 위치 (num_bodies, 3)
        mocap_body_pos: 목표 바디 위치 (num_bodies, 3)
        robot_body_quat: 로봇 바디 쿼터니언 (num_bodies, 4)
        mocap_body_quat: 목표 바디 쿼터니언 (num_bodies, 4)
        robot_body_lin_vel: 로봇 바디 선형 속도 (num_bodies, 3)
        mocap_body_lin_vel: 목표 바디 선형 속도 (num_bodies, 3)
        robot_body_ang_vel: 로봇 바디 각속도 (num_bodies, 3)
        mocap_body_ang_vel: 목표 바디 각속도 (num_bodies, 3)
    
    Returns:
        dict: 추가 성능 지표들
    """
    metrics = {}
    
    # 1. 앵커 바디 추적 성능 (commands.py 기반)
    # 앵커 바디(torso_link)의 위치 오차 - 논문의 ξ_{b_anchor} 위치 부분
    # 로봇과 목표 모션 간의 앵커 바디 위치 차이를 유클리드 거리로 계산 (단위: m)
    metrics['error_anchor_body_pos'] = np.linalg.norm(robot_anchor_pos_w - mocap_anchor_pos_w)
    
    # 앵커 바디(torso_link)의 회전 오차 - 논문의 ξ_{b_anchor} 회전 부분
    # 로봇과 목표 모션 간의 앵커 바디 자세 차이를 쿼터니언 오차로 계산 (단위: rad)
    metrics['error_anchor_body_rot'] = quat_error_magnitude(robot_anchor_quat, mocap_anchor_quat)
    
    # 2. 관절 추적 성능 (commands.py 기반)
    # 전체 관절의 위치 오차 - 모든 관절의 위치 차이를 유클리드 거리로 계산 (단위: rad)
    # 이는 로봇이 목표 모션의 관절 위치를 얼마나 정확히 따라가는지를 측정
    metrics['error_joint_pos'] = np.linalg.norm(robot_joint_pos - mocap_joint_pos)
    
    # 전체 관절의 속도 오차 - 모든 관절의 속도 차이를 유클리드 거리로 계산 (단위: rad/s)
    # 이는 로봇이 목표 모션의 관절 속도를 얼마나 정확히 따라가는지를 측정
    metrics['error_joint_vel'] = np.linalg.norm(robot_joint_vel - mocap_joint_vel)
    
    # 3. 바디 부위 추적 성능 (commands.py 기반)
    # 대표 바디들(손목, 발목 등)의 추적 성능을 평가
    if robot_body_pos is not None and mocap_body_pos is not None:
        # 대표 바디들의 위치 오차 - 각 바디별 위치 차이의 평균 (단위: m)
        # 손목, 발목 등 주요 부위가 목표 모션과 얼마나 일치하는지 측정
        metrics['error_non_anchor_body_pos'] = np.mean(np.linalg.norm(robot_body_pos - mocap_body_pos, axis=-1))
        
        # 대표 바디들의 회전 오차 - 각 바디별 자세 차이의 평균 (단위: rad)
        # 손목, 발목 등 주요 부위의 자세가 목표 모션과 얼마나 일치하는지 측정
        metrics['error_non_anchor_body_rot'] = np.mean([quat_error_magnitude(robot_body_quat[i], mocap_body_quat[i]) 
                                           for i in range(len(robot_body_quat))])
        
        # # 대표 바디들의 선형 속도 오차 - 각 바디별 선형 속도 차이의 평균 (단위: m/s)
        # # 현재는 실제 속도 데이터가 없어서 0으로 설정됨
        # metrics['error_body_lin_vel'] = np.mean(np.linalg.norm(robot_body_lin_vel_w - mocap_body_lin_vel_w, axis=-1))
        
        # # 대표 바디들의 각속도 오차 - 각 바디별 각속도 차이의 평균 (단위: rad/s)
        # # 현재는 실제 속도 데이터가 없어서 0으로 설정됨
        # metrics['error_body_ang_vel'] = np.mean(np.linalg.norm(robot_body_ang_vel_w - mocap_body_ang_vel_w, axis=-1))
    
    return metrics

def save_performance_plots(additional_metrics, save_dir="/home/keti/whole_body_tracking/scripts/Beyond_mimic_sim2sim_G1/performance_plots", simulation_dt=0.005, control_decimation=4):
    """
    commands.py 기반 성능 지표들을 시각화하고 저장합니다.
    
    Args:
        additional_metrics: commands.py 기반 성능 지표 데이터
        save_dir: 저장할 디렉토리
        simulation_dt: 시뮬레이션 타임스텝 (초)
        control_decimation: 제어기 업데이트 주파수 (시뮬레이션 스텝 대비)
    """
    # 저장 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 시간 축 계산 (제어기 업데이트 주파수 기준)
    control_dt = simulation_dt * control_decimation  # 0.005 * 4 = 0.02초 (50Hz)
    if additional_metrics and 'error_anchor_body_pos' in additional_metrics:
        num_steps = len(additional_metrics['error_anchor_body_pos'])
        time_axis = np.arange(num_steps) * control_dt  # 시간 축 (초)
    else:
        time_axis = None
    
    # 1. 앵커 및 관절 성능 지표 플롯
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Sim-to-Sim Deploy Performance Metrics (commands.py based)', fontsize=16)
    
    # 앵커 위치 오차
    if 'error_anchor_body_pos' in additional_metrics and additional_metrics['error_anchor_body_pos']:
        axes[0, 0].plot(time_axis, additional_metrics['error_anchor_body_pos'], 'b-', linewidth=1)
        axes[0, 0].set_title('Anchor Body Position Error')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Error (m)')
        axes[0, 0].grid(True)
    
    # 앵커 회전 오차
    if 'error_anchor_body_rot' in additional_metrics and additional_metrics['error_anchor_body_rot']:
        axes[0, 1].plot(time_axis, additional_metrics['error_anchor_body_rot'], 'r-', linewidth=1)
        axes[0, 1].set_title('Anchor Body Rotation Error')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Error (rad)')
        axes[0, 1].grid(True)
    
    # 관절 위치 오차
    if 'error_joint_pos' in additional_metrics and additional_metrics['error_joint_pos']:
        axes[1, 0].plot(time_axis, additional_metrics['error_joint_pos'], 'g-', linewidth=1)
        axes[1, 0].set_title('Joint Position Error')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Error (rad)')
        axes[1, 0].grid(True)
    
    # 관절 속도 오차
    if 'error_joint_vel' in additional_metrics and additional_metrics['error_joint_vel']:
        axes[1, 1].plot(time_axis, additional_metrics['error_joint_vel'], 'm-', linewidth=1)
        axes[1, 1].set_title('Joint Velocity Error')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Error (rad/s)')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/anchor_joint_metrics_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 바디 부위별 성능 플롯 (위치와 회전만)
    if ('error_non_anchor_body_pos' in additional_metrics and additional_metrics['error_non_anchor_body_pos'] and
        'error_non_anchor_body_rot' in additional_metrics and additional_metrics['error_non_anchor_body_rot']):
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Non-Anchor Body Part Tracking Performance (commands.py based)', fontsize=16)
        
        # 바디 위치 오차
        axes[0].plot(time_axis, additional_metrics['error_non_anchor_body_pos'], 'b-', linewidth=1)
        axes[0].set_title('Non-Anchor Body Position Error')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Error (m)')
        axes[0].grid(True)
        
        # 바디 회전 오차
        axes[1].plot(time_axis, additional_metrics['error_non_anchor_body_rot'], 'r-', linewidth=1)
        axes[1].set_title('Non-Anchor Body Rotation Error')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Error (rad)')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/non_anchor_body_metrics_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"📊 성능 플롯이 저장되었습니다: {save_dir}")
    print(f"   - 앵커/관절 성능: anchor_joint_metrics_{timestamp}.png")
    if ('error_non_anchor_body_pos' in additional_metrics and additional_metrics['error_non_anchor_body_pos']):
        print(f"   - Non Anchor Body 부위 성능: non_anchor_body_metrics_{timestamp}.png")



def initialize_additional_metrics():
    """성능 지표 초기화 (commands.py 기반) (Plot 시 사용)
    
    Returns:
        dict: 초기화된 성능 지표 딕셔너리
    """
    return {
        'error_anchor_body_pos': [],
        'error_anchor_body_rot': [],
        'error_joint_pos': [],
        'error_joint_vel': [],
        'error_non_anchor_body_pos': [],
        'error_non_anchor_body_rot': [],
        # 'error_body_lin_vel': [],
        # 'error_body_ang_vel': []
    }


def initialize_tracking_metrics(mj_model, anchor_body_name, isaac_body_names):
    """트래킹 성능 로깅을 위한 모든 변수들 초기화
    
    Args:
        mj_model: MuJoCo 모델
        anchor_body_name: 앵커 body 이름
        isaac_body_names: Isaac Lab body 이름 리스트
        
    Returns:
        tuple: (mujoco_anchor_body_id, representative_body_ids, isaac_representative_body_ids, additional_metrics)
    """
    # mujoco에서 body 이름으로 body id 찾기 : 16번째 body의 이름은 torso_link 이다.
    mujoco_anchor_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, anchor_body_name)
    if mujoco_anchor_body_id == -1:
        raise ValueError(f"Body {anchor_body_name} not found in model")
    
    # 대표 body들 정의 및 ID 매핑
    representative_bodies = get_representative_bodies()
    representative_body_ids = get_mujoco_body_ids(mj_model, representative_bodies)
    isaac_representative_body_ids = get_isaac_body_indices(isaac_body_names, representative_bodies)
    
    # 성능 지표 초기화
    additional_metrics = initialize_additional_metrics()
    
    return mujoco_anchor_body_id, representative_body_ids, isaac_representative_body_ids, additional_metrics

def calculate_and_log_performance_metrics(print_log,timestep, mj_data, mocap_joint_pos, mocap_joint_vel, 
                                        mocap_pos, mocap_quat, robot_anchor_pos, robot_anchor_quat,
                                        mocap_anchor_pos, mocap_anchor_quat, additional_metrics,
                                        log_interval, mujoco_joint_seq, isaac_joint_seq,
                                        representative_body_ids, isaac_representative_body_ids,
                                        representative_bodies):
    """
    성능 지표 계산 및 로깅을 담당하는 함수
    
    Args:
        timestep: 현재 시간 스텝
        mj_data: MuJoCo 시뮬레이션 데이터
        mocap_joint_pos: 모션 캡처 관절 위치 데이터
        mocap_joint_vel: 모션 캡처 관절 속도 데이터
        mocap_pos: 모션 캡처 바디 위치 데이터
        mocap_quat: 모션 캡처 바디 자세 데이터
        robot_anchor_pos: 로봇 앵커 위치
        robot_anchor_quat: 로봇 앵커 자세
        mocap_anchor_pos: 모션 캡처 앵커 위치
        mocap_anchor_quat: 모션 캡처 앵커 자세
        additional_metrics: 성능 지표 저장 딕셔너리
        log_interval: 로깅 간격
        mujoco_joint_seq: MuJoCo 관절 순서
        isaac_joint_seq: Isaac Lab 관절 순서
        representative_body_ids: 대표 바디 ID 매핑
        isaac_representative_body_ids: Isaac Lab 바디 인덱스 매핑
        representative_bodies: 대표 바디 정보
    """
    # 관절 데이터 수집 (Isaac Lab 순서로 변환)
    current_joint_pos = mj_data.qpos[7:]  # 현재 관절 위치 (MuJoCo 순서)
    target_joint_pos_isaac = mocap_joint_pos[timestep, :]  # 목표 관절 위치 (Isaac 순서)
    current_joint_pos_isaac = np.array([current_joint_pos[mujoco_joint_seq.index(joint)] for joint in isaac_joint_seq])
    
    current_joint_vel = mj_data.qvel[6:]  # 현재 관절 속도 (MuJoCo 순서)
    target_joint_vel_isaac = mocap_joint_vel[timestep, :]  # 목표 관절 속도 (Isaac 순서)
    current_joint_vel_isaac = np.array([current_joint_vel[mujoco_joint_seq.index(joint)] for joint in isaac_joint_seq])
    
    # 바디 부위 데이터 수집 (대표 body들만)
    robot_body_pos = np.array([mj_data.xpos[representative_body_ids[key]] for key in representative_bodies.keys() 
                             if key in representative_body_ids])
    mocap_body_pos = np.array([mocap_pos[timestep, isaac_representative_body_ids[key], :] for key in representative_bodies.keys() 
                             if key in isaac_representative_body_ids])
    robot_body_quat = np.array([mj_data.xquat[representative_body_ids[key]] for key in representative_bodies.keys() 
                              if key in representative_body_ids])
    mocap_body_quat = np.array([mocap_quat[timestep, isaac_representative_body_ids[key], :] for key in representative_bodies.keys() 
                              if key in isaac_representative_body_ids])
    
    # 바디 속도 데이터 (간단히 0으로 설정 - 실제로는 이전 프레임과의 차이로 계산 가능)
    robot_body_lin_vel = np.zeros_like(robot_body_pos)
    mocap_body_lin_vel = np.zeros_like(mocap_body_pos)
    robot_body_ang_vel = np.zeros_like(robot_body_pos)
    mocap_body_ang_vel = np.zeros_like(mocap_body_pos)
    
    # 성능 지표 계산 (commands.py 기반)
    additional_metrics_step = calculate_additional_metrics(
        robot_anchor_pos_w=robot_anchor_pos,
        robot_anchor_quat=robot_anchor_quat,
        mocap_anchor_pos_w=mocap_anchor_pos,
        mocap_anchor_quat=mocap_anchor_quat,
        robot_joint_pos=current_joint_pos_isaac,
        mocap_joint_pos=target_joint_pos_isaac,
        robot_joint_vel=current_joint_vel_isaac,
        mocap_joint_vel=target_joint_vel_isaac,
        robot_body_pos=robot_body_pos,
        mocap_body_pos=mocap_body_pos,
        robot_body_quat=robot_body_quat,
        mocap_body_quat=mocap_body_quat,
        robot_body_lin_vel_w=robot_body_lin_vel,
        mocap_body_lin_vel_w=mocap_body_lin_vel,
        robot_body_ang_vel_w=robot_body_ang_vel,
        mocap_body_ang_vel_w=mocap_body_ang_vel
    )
    
    if print_log==True:
        # 성능 지표 저장
        for key, value in additional_metrics_step.items():
            additional_metrics[key].append(value)
        
        # 실시간 로깅 출력 (commands.py 기반 지표 사용)
        if timestep % log_interval == 0:
            print(f"\n=== 트래킹 성능 리포트 (Step {timestep}) ===")
            print(f"Anchor Position Error: {additional_metrics_step['error_anchor_body_pos']:.4f} m")
            print(f"Anchor Rotation Error: {additional_metrics_step['error_anchor_body_rot']:.4f} rad")
            print(f"Joint Position Error: {additional_metrics_step['error_joint_pos']:.4f} rad")
            print(f"Joint Velocity Error: {additional_metrics_step['error_joint_vel']:.4f} rad/s")
            
            if 'error_non_anchor_body_pos' in additional_metrics_step:
                print(f"Body Position Error: {additional_metrics_step['error_non_anchor_body_pos']:.4f} m")
                print(f"Body Rotation Error: {additional_metrics_step['error_non_anchor_body_rot']:.4f} rad")
            
            # 최근 100스텝 평균 성능
            if len(additional_metrics['error_anchor_body_pos']) >= log_interval:
                recent_anchor_pos = np.mean(additional_metrics['error_anchor_body_pos'][-log_interval:])
                recent_anchor_rot = np.mean(additional_metrics['error_anchor_body_rot'][-log_interval:])
                recent_joint_pos = np.mean(additional_metrics['error_joint_pos'][-log_interval:])
                recent_joint_vel = np.mean(additional_metrics['error_joint_vel'][-log_interval:])
                
                print(f"\n최근 {log_interval}스텝 평균:")
                print(f"   Anchor Position: {recent_anchor_pos:.4f} m")
                print(f"   Anchor Rotation: {recent_anchor_rot:.4f} rad")
                print(f"   Joint Position: {recent_joint_pos:.4f} rad")
                print(f"   Joint Velocity: {recent_joint_vel:.4f} rad/s")
                
                if 'error_non_anchor_body_pos' in additional_metrics and len(additional_metrics['error_non_anchor_body_pos']) >= log_interval:
                    recent_body_pos = np.mean(additional_metrics['error_non_anchor_body_pos'][-log_interval:])
                    recent_body_rot = np.mean(additional_metrics['error_non_anchor_body_rot'][-log_interval:])
                    print(f"   Body Position: {recent_body_pos:.4f} m")
                    print(f"   Body Rotation: {recent_body_rot:.4f} rad")

