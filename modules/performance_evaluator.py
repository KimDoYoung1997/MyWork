"""
Performance evaluation and result reporting module
This module provides functions to evaluate and report simulation performance metrics
"""

import numpy as np
from typing import Dict, List, Any

def evaluate_simulation_performance(additional_metrics: Dict[str, List[float]], 
                                  simulation_dt: float, 
                                  control_decimation: int) -> Dict[str, Any]:
    """Simulation performance evaluation based on commands.py metrics
    
    Args:
        additional_metrics: Dictionary containing performance metrics
        simulation_dt: Simulation timestep
        control_decimation: Control decimation factor
        
    Returns:
        Dictionary containing evaluated performance metrics
    """
    # commands.py 기반 핵심 성능 지표 계산
    avg_anchor_body_pos_error = np.mean(additional_metrics['error_anchor_body_pos'])
    avg_anchor_body_rot_error = np.mean(additional_metrics['error_anchor_body_rot'])
    avg_joint_pos_error = np.mean(additional_metrics['error_joint_pos'])
    avg_joint_vel_error = np.mean(additional_metrics['error_joint_vel'])
    
    max_anchor_body_pos_error = np.max(additional_metrics['error_anchor_body_pos'])
    max_anchor_body_rot_error = np.max(additional_metrics['error_anchor_body_rot'])
    
    # 바디 부위 성능 (있는 경우)
    body_performance = {}
    if 'error_non_anchor_body_pos' in additional_metrics and additional_metrics['error_non_anchor_body_pos']:
        body_performance['avg_body_pos_error'] = np.mean(additional_metrics['error_non_anchor_body_pos'])
        body_performance['avg_body_rot_error'] = np.mean(additional_metrics['error_non_anchor_body_rot'])
    
    # 시뮬레이션 통계
    total_steps = len(additional_metrics['error_anchor_body_pos'])
    simulation_time = total_steps * simulation_dt
    policy_frequency = 1 / (simulation_dt * control_decimation)
    
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
    """Print comprehensive performance report
    
    Args:
        performance_metrics: Evaluated performance metrics
        additional_metrics: Raw performance metrics for body parts
    """
    print(f"commands.py 기반 핵심 성능 지표:")
    print(f"   Anchor Position Error: {performance_metrics['avg_anchor_pos_error']:.4f} m (최대: {performance_metrics['max_anchor_pos_error']:.4f} m)")
    print(f"   Anchor Rotation Error: {performance_metrics['avg_anchor_rot_error']:.4f} rad (최대: {performance_metrics['max_anchor_rot_error']:.4f} rad)")
    print(f"   Joint Position Error: {performance_metrics['avg_joint_pos_error']:.4f} rad")
    print(f"   Joint Velocity Error: {performance_metrics['avg_joint_vel_error']:.4f} rad/s")
    
    # 바디 부위 성능 (있는 경우)
    if 'avg_body_pos_error' in performance_metrics:
        print(f"\nBody Part Performance:")
        print(f"   Body Position Error: {performance_metrics['avg_body_pos_error']:.4f} m")
        print(f"   Body Rotation Error: {performance_metrics['avg_body_rot_error']:.4f} rad")
    
    print(f"\nSim-to-Sim 실행 통계:")
    print(f"   총 처리된 스텝: {performance_metrics['total_steps']}")
    print(f"   시뮬레이션 시간: {performance_metrics['simulation_time']:.2f}초")
    print(f"   정책 추론 주파수: {performance_metrics['policy_frequency']:.1f}Hz")

def evaluate_sim2sim_success(performance_metrics: Dict[str, float]) -> str:
    """Evaluate Sim-to-Sim success level based on performance metrics
    
    Args:
        performance_metrics: Evaluated performance metrics
        
    Returns:
        Success level string
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
    """Print Sim-to-Sim success report based on success level
    
    Args:
        success_level: Success level string
    """
    if success_level == "excellent":
        print("\n🎉 Sim-to-Sim 성공도: 우수 (Excellent)")
        print("   Beyond Mimic 방법론이 성공적으로 구현되었습니다!")
        print("   Isaac Lab → MuJoCo 전환이 매우 정확하게 수행되었습니다.")
    elif success_level == "good":
        print("\n Sim-to-Sim 성공도: 양호 (Good)")
        print("   모션 트래킹이 잘 수행되고 있지만 개선 여지가 있습니다.")
        print("   좌표계 변환 없이도 상당한 성능을 달성했습니다.")
    else:
        print("\n⚠️  Sim-to-Sim 성공도: 개선 필요 (Needs Improvement)")
        print("   정책 튜닝이나 학습 데이터 개선이 필요할 수 있습니다.")
        print("   앵커링 메커니즘이나 관찰값 구성 재검토를 권장합니다.")

def generate_final_performance_report(additional_metrics: Dict[str, List[float]], 
                                   simulation_dt: float, 
                                   control_decimation: int) -> tuple:
    """Generate and print complete performance report
    
    Args:
        additional_metrics: Raw performance metrics
        simulation_dt: Simulation timestep
        control_decimation: Control decimation factor
    """
    # 성능 지표 평가
    performance_metrics = evaluate_simulation_performance(
        additional_metrics, simulation_dt, control_decimation
    )
    
    # 성능 리포트 출력
    print_performance_report(performance_metrics, additional_metrics)
    
    # Sim-to-Sim 성공도 평가 및 출력
    success_level = evaluate_sim2sim_success(performance_metrics)
    print_sim2sim_success_report(success_level)
    
    return performance_metrics, success_level
