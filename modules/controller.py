def pd_control(target_q, current_q, kp, target_dq, current_dq, kd):
    """Calculates torques from position commands"""
    return (target_q - current_q) * kp + (target_dq - current_dq) * kd

