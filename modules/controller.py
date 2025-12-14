def pd_control(target_q, current_q, kp, target_dq, current_dq, kd):
    """PD 제어기를 사용하여 목표 관절 위치/속도로부터 제어 토크를 계산
    
    PD (Proportional-Derivative) 제어 공식:
    τ = kp * (target_q - current_q) + kd * (target_dq - current_dq)
    
    Args:
        target_q: 목표 관절 위치 (29,) [rad]
        current_q: 현재 관절 위치 (29,) [rad]
        kp: 비례 제어 게인 (stiffness) (29,)
        target_dq: 목표 관절 속도 (29,) [rad/s]
        current_dq: 현재 관절 속도 (29,) [rad/s]
        kd: 미분 제어 게인 (damping) (29,)
    
    Returns:
        τ: 계산된 제어 토크 (29,) [N⋅m]
    """
    return (target_q - current_q) * kp + (target_dq - current_dq) * kd

