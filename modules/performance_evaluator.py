"""
Performance evaluation and result reporting module
This module provides functions to evaluate and report simulation performance metrics
"""

import numpy as np
from typing import Dict, List, Any

def evaluate_simulation_performance(additional_metrics: Dict[str, List[float]], 
                                  simulation_dt: float, 
                                  control_decimation: int) -> Dict[str, Any]:
    """시뮬레이션 전체 성능을 평가하여 통계 계산
    
    시계열 성능 지표 데이터로부터 평균, 최대값 등의 통계를 계산하여
    전반적인 모션 트래킹 성능을 요약합니다.
    
    Args:
        additional_metrics: 성능 지표 시계열 데이터 (각 키는 시간에 따른 값의 리스트)
        simulation_dt: 시뮬레이션 타임스텝 (초, 예: 0.005)
        control_decimation: 제어 decimation (예: 4)
        
    Returns:
        평가된 성능 지표 딕셔너리:
            - avg_anchor_pos_error: Anchor 위치 오차 평균 [m]
            - avg_anchor_rot_error: Anchor 회전 오차 평균 [rad]
            - avg_joint_pos_error: 관절 위치 오차 평균 [rad]
            - avg_joint_vel_error: 관절 속도 오차 평균 [rad/s]
            - max_anchor_pos_error: Anchor 위치 오차 최대값 [m]
            - max_anchor_rot_error: Anchor 회전 오차 최대값 [rad]
            - total_steps: 전체 제어 스텝 수
            - simulation_time: 시뮬레이션 시간 [s]
            - policy_frequency: 정책 추론 주파수 [Hz]
    """
    # 핵심 성능 지표의 평균값 계산
    avg_anchor_body_pos_error = np.mean(additional_metrics['error_anchor_body_pos'])  # Anchor 위치 오차 평균 [m]
    avg_anchor_body_rot_error = np.mean(additional_metrics['error_anchor_body_rot'])  # Anchor 회전 오차 평균 [rad]
    avg_joint_pos_error = np.mean(additional_metrics['error_joint_pos'])  # 관절 위치 오차 평균 [rad]
    avg_joint_vel_error = np.mean(additional_metrics['error_joint_vel'])  # 관절 속도 오차 평균 [rad/s]
    
    # 핵심 성능 지표의 최대값 계산 (worst-case 분석)
    max_anchor_body_pos_error = np.max(additional_metrics['error_anchor_body_pos'])  # [m]
    max_anchor_body_rot_error = np.max(additional_metrics['error_anchor_body_rot'])  # [rad]
    
    # 대표 바디(손목, 발목) 성능 (데이터가 있는 경우)
    body_performance = {}
    if 'error_non_anchor_body_pos' in additional_metrics and additional_metrics['error_non_anchor_body_pos']:
        body_performance['avg_body_pos_error'] = np.mean(additional_metrics['error_non_anchor_body_pos'])  # [m]
        body_performance['avg_body_rot_error'] = np.mean(additional_metrics['error_non_anchor_body_rot'])  # [rad]
    
    # 시뮬레이션 통계 정보
    total_steps = len(additional_metrics['error_anchor_body_pos'])  # 제어 스텝 수 (50Hz 기준)
    simulation_time = total_steps * simulation_dt * control_decimation  # 실제 시뮬레이션 시간 [s]
    policy_frequency = 1 / (simulation_dt * control_decimation)  # 정책 추론 주파수 [Hz] (보통 50Hz)
    
    return {
        'avg_anchor_pos_error': avg_anchor_body_pos_error,
        'avg_anchor_rot_error': avg_anchor_body_rot_error,
        'avg_joint_pos_error': avg_joint_pos_error,
        'avg_joint_vel_error': avg_joint_vel_error,
        'max_anchor_pos_error': max_anchor_body_pos_error,
        'max_anchor_rot_error': max_anchor_body_rot_error,
        'total_steps': total_steps,
        'simulation_time': simulation_time,
        'policy_frequency': policy_frequency,
        **body_performance
    }

def print_performance_report(performance_metrics: Dict[str, float], 
                           additional_metrics: Dict[str, List[float]]) -> None:
    """종합 성능 리포트를 콘솔에 출력
    
    Args:
        performance_metrics: 평가된 성능 지표 (evaluate_simulation_performance() 반환값)
        additional_metrics: 원본 성능 지표 시계열 (현재 미사용, 향후 확장 가능)
    """
    print(f"Beyond Mimic 모션 트래킹 성능 지표:")
    print(f"   Anchor Position Error: {performance_metrics['avg_anchor_pos_error']:.4f} m (최대: {performance_metrics['max_anchor_pos_error']:.4f} m)")
    print(f"   Anchor Rotation Error: {performance_metrics['avg_anchor_rot_error']:.4f} rad (최대: {performance_metrics['max_anchor_rot_error']:.4f} rad)")
    print(f"   Joint Position Error: {performance_metrics['avg_joint_pos_error']:.4f} rad")
    print(f"   Joint Velocity Error: {performance_metrics['avg_joint_vel_error']:.4f} rad/s")
    
    # 대표 바디(손목, 발목) 성능 (데이터가 있는 경우)
    if 'avg_body_pos_error' in performance_metrics:
        print(f"\n대표 Body(손목, 발목) 추적 성능:")
        print(f"   Body Position Error: {performance_metrics['avg_body_pos_error']:.4f} m")
        print(f"   Body Rotation Error: {performance_metrics['avg_body_rot_error']:.4f} rad")
    
    print(f"\nSim-to-Sim 실행 통계:")
    print(f"   총 제어 스텝: {performance_metrics['total_steps']}")
    print(f"   시뮬레이션 시간: {performance_metrics['simulation_time']:.2f}초")
    print(f"   정책 추론 주파수: {performance_metrics['policy_frequency']:.1f}Hz")

def evaluate_sim2sim_success(performance_metrics: Dict[str, float]) -> str:
    """Sim-to-Sim 전환 성공도를 평가
    
    Anchor body의 추적 오차를 기준으로 모션 트래킹 품질을 3단계로 평가합니다.
    
    Args:
        performance_metrics: 평가된 성능 지표
        
    Returns:
        성공도 수준: "excellent", "good", "needs_improvement"
        
    평가 기준:
        - excellent: avg_anchor_pos < 0.01m AND avg_anchor_rot < 0.1rad
        - good: avg_anchor_pos < 0.05m AND avg_anchor_rot < 0.3rad
        - needs_improvement: 위 기준 미달
    """
    avg_anchor_pos_error = performance_metrics['avg_anchor_pos_error']
    avg_anchor_rot_error = performance_metrics['avg_anchor_rot_error']
    
    if avg_anchor_pos_error < 0.01 and avg_anchor_rot_error < 0.1:
        return "excellent"
    elif avg_anchor_pos_error < 0.05 and avg_anchor_rot_error < 0.3:
        return "good"
    else:
        return "needs_improvement"

def print_sim2sim_success_report(success_level: str) -> None:
    """Sim-to-Sim 성공도 리포트를 출력
    
    Args:
        success_level: 성공도 수준 ("excellent", "good", "needs_improvement")
    """
    if success_level == "excellent":
        print("\n🎉 Sim-to-Sim 성공도: 우수 (Excellent)")
        print("   Beyond Mimic 방법론이 성공적으로 구현되었습니다!")
        print("   Isaac Lab → MuJoCo 전환이 매우 정확하게 수행되었습니다.")
        print("   로봇이 reference motion을 정밀하게 추적하고 있습니다.")
    elif success_level == "good":
        print("\n✓ Sim-to-Sim 성공도: 양호 (Good)")
        print("   모션 트래킹이 잘 수행되고 있지만 개선 여지가 있습니다.")
        print("   좌표계 차이(Z-up vs Y-up)에도 불구하고 상대 변환 기반으로 작동합니다.")
        print("   추가 학습 또는 파라미터 튜닝으로 성능 향상 가능합니다.")
    else:
        print("\n⚠️  Sim-to-Sim 성공도: 개선 필요 (Needs Improvement)")
        print("   모션 트래킹 성능이 기대에 미치지 못합니다.")
        print("   가능한 원인:")
        print("   - 학습 데이터의 quality 문제")
        print("   - 정책 학습이 불충분 (더 많은 iteration 필요)")
        print("   - PD 제어 게인(stiffness/damping) 튜닝 필요")
        print("   - Observation 구성 확인 (SE vs woSE)")

def generate_final_performance_report(additional_metrics: Dict[str, List[float]], 
                                   simulation_dt: float, 
                                   control_decimation: int) -> tuple:
    """최종 성능 리포트 생성 및 출력 (통합 함수)
    
    시계열 성능 데이터로부터 통계를 계산하고, 
    종합 리포트와 성공도 평가를 콘솔에 출력합니다.
    
    Args:
        additional_metrics: 원본 성능 지표 시계열 데이터
        simulation_dt: 시뮬레이션 타임스텝 (초)
        control_decimation: 제어 decimation factor
        
    Returns:
        tuple: (performance_metrics, success_level)
            - performance_metrics: 평가된 성능 지표 딕셔너리
            - success_level: 성공도 수준 문자열
    """
    # 1. 성능 지표 통계 계산
    performance_metrics = evaluate_simulation_performance(
        additional_metrics, simulation_dt, control_decimation
    )
    
    # 2. 종합 성능 리포트 출력
    print_performance_report(performance_metrics, additional_metrics)
    
    # 3. Sim-to-Sim 성공도 평가 및 출력
    success_level = evaluate_sim2sim_success(performance_metrics)
    print_sim2sim_success_report(success_level)
    
    return performance_metrics, success_level
