#  좌표계 변환 분석: motion_anchor_pos_b vs base_lin_vel

##  핵심 질문

> **"base_lin_vel을 좌표계 변환했는데, motion_anchor_pos_b는 왜 안 바꿔도 되나?"**

---

##  답변: 둘은 다른 종류의 변환이다.

### **1. motion_anchor_pos_b: 상대 변환 (Relative Transform)**

#### **무엇을 계산하는가?**
```
"로봇을 기준으로 봤을 때, 목표 위치가 어디에 있는가?"
```

#### **수학적 정의**
```python
# compute_relative_transform_mujoco() 내부
T_A = pose_to_transformation_matrix(robot_pos, robot_quat)  # Robot world pose
T_B = pose_to_transformation_matrix(mocap_pos, mocap_quat)  # Mocap world pose
T_rel = T_A^(-1) * T_B  # Relative transform
rel_pos = T_rel[0:3, 3]  # Already in robot body frame!
```

#### **왜 이미 올바른가?**
- `T_A^(-1)`: Robot world frame → Robot body frame 변환
- `T_rel`: Robot body frame 기준의 상대 변환
- **결과: rel_pos는 이미 robot body frame에서 표현됨!**

#### **예시**
```
World frame:
- Robot position: [0, 0, 1]
- Mocap position: [1, 0, 1]

Robot body frame (robot을 원점으로):
- Robot position: [0, 0, 0]  ← robot이 기준점
- Mocap position: [1, 0, 0]  ← robot 기준 상대 위치

→ motion_anchor_pos_b = [1, 0, 0] (robot body frame)
```

---

### **2. base_lin_vel: 벡터 회전 (Vector Rotation)**

#### **무엇을 계산하는가?**
```
"World frame에서 측정된 속도를 Robot body frame에서 어떻게 보이는가?"
```

#### **수학적 정의**
```python
# transform_velocity_to_body_frame() 내부
v_world = qvel[0:3]  # Velocity in world frame
v_body = R^T * v_world  # Rotate to body frame
```
- `R`: World → Body 회전 행렬
- `R^T`: 역회전 (Body → World의 역)

#### **왜 변환이 필요한가?**
- 속도는 **절대 벡터** (방향과 크기를 가짐)
- World frame의 [1, 0, 0]은 "world의 x축 방향"
- Robot body frame의 [1, 0, 0]은 "robot의 전방 방향"
- **같은 속도라도 frame에 따라 다르게 표현됨!**

#### **예시**
```
World frame:
- Robot facing: East (90° rotated)
- Velocity: [1, 0, 0] (North)

Robot body frame:
- Robot facing: Forward
- Same velocity: [0, -1, 0] (Left)

→ World [1, 0, 0] ≠ Body [1, 0, 0]
→ 회전 변환 필요!
```

---

##  비교 표

| 요소 | motion_anchor_pos_b | base_lin_vel |
|-----|-------------------|-------------|
| **타입** | 상대 변환 | 벡터 회전 |
| **입력** | 두 개의 world pose | 하나의 world velocity |
| **연산** | T_A^(-1) * T_B | R^T * v |
| **좌표계 변환** |  포함됨 (T_A^(-1)) |  별도 필요 |
| **결과 frame** | Robot body frame | World frame → Body frame |
| **함수** | `compute_relative_transform_mujoco()` | `transform_velocity_to_body_frame()` |

---

##  왜 혼란스러웠나?

### **공통점**
- 둘 다 "robot body frame"에서 표현되어야 함
- 둘 다 world frame 데이터로부터 계산

### **차이점**
- **motion_anchor_pos_b**: 
  - **두 pose 간의 상대 관계** 계산
  - 상대 변환 자체가 좌표계 변환을 포함
  
- **base_lin_vel**:
  - **한 벡터의 표현** 변환
  - 명시적 회전 변환 필요

---

##  핵심 통찰

### **1. 상대 위치 vs 절대 벡터**

```python
# 상대 위치 (motion_anchor_pos_b)
"A에서 B로 가려면 어느 방향으로 가야 하나?"
→ 자동으로 A 기준 좌표계

# 절대 벡터 (base_lin_vel)
"현재 속도가 [1, 0, 0]이다"
→ 어느 좌표계? 명시 필요!
```

### **2. 변환의 종류**

```python
# Transformation matrix (4x4)
T = [R  t]  # R: rotation, t: translation
    [0  1]

# Relative transform
T_rel = T_A^(-1) * T_B  # Includes coordinate change

# Vector rotation
v_body = R^T * v_world  # Only rotation, no translation
```

---

##  구현 상세

### **motion_anchor_pos_b 계산 과정**

```python
# Line 341-346 in main.py
anchor_pos_track_error, temp_anchor_quat_track_error = compute_relative_transform_mujoco(
    mujoco_robot_anchor_pos_A=mujoco_robot_anchor_pos,    # World frame
    mujoco_robot_anchor_quat_A=mujoco_robot_anchor_quat,  # World frame
    isaac_ref_pos_B=mocap_anchor_pos,                     # World frame
    isaac_ref_quat_B=mocap_anchor_quat                    # World frame
)
# → anchor_pos_track_error는 이미 robot body frame!
```

**내부 연산:**
```python
T_A = [R_A  p_A]  # Robot world pose
      [0    1  ]
      
T_B = [R_B  p_B]  # Mocap world pose
      [0    1  ]
      
T_rel = T_A^(-1) * T_B = [R_A^T    -R_A^T*p_A] * [R_B  p_B]
                          [0        1         ]   [0    1  ]
                          
                        = [R_A^T*R_B    R_A^T*(p_B - p_A)]
                          [0            1                ]
                          
rel_pos = R_A^T * (p_B - p_A)  # Body frame에서 표현된 상대 위치!
```

### **base_lin_vel 계산 과정**

```python
# Line 376-387 in main.py
world_lin_vel = mj_data.qvel[0:3]  # World frame
robot_quat = mj_data.xquat[mujoco_anchor_body_id]

root_lin_vel = transform_velocity_to_body_frame(world_lin_vel, robot_quat)
# → root_lin_vel은 이제 robot body frame!
```

**내부 연산:**
```python
R = quat_to_rotation_matrix(robot_quat)  # World → Body rotation
v_body = R^T * v_world  # Rotate velocity vector
```

---

##  Summary

### **1. "Body frame으로 변환"의 두 가지 의미**

#### **A. 상대 변환 (Relative Transform)**
- 기준점을 바꾸는 것
- "A 기준으로 B가 어디에 있나?"
- Transformation matrix 연산

#### **B. 벡터 회전 (Vector Rotation)**
- 벡터의 표현을 바꾸는 것
- "이 방향을 다른 좌표계에서 어떻게 표현하나?"
- Rotation matrix 연산

### **2. 자동 vs 명시적 변환**

```python
# 자동 포함 (상대 변환)
T_rel = T_A^(-1) * T_B  #  좌표계 변환 자동 포함

# 명시적 필요 (벡터 회전)
v_body = transform_velocity_to_body_frame(v_world, quat)  #  명시적 호출 필요
```

### **3. 논문 구현의 미묘함**

```
논문: "V_b_root expressed in root frame"
```

- **위치 error**: 이미 body frame (상대 변환 특성)
- **속도**: 명시적 변환 필요 (벡터 회전)

---

##  최종 구현 정리

### **transforms.py에 추가된 함수**

```python
def transform_velocity_to_body_frame(world_vel, body_quat):
    """World frame velocity → Body frame velocity
    
    Args:
        world_vel: (3,) velocity in world frame
        body_quat: (4,) body orientation [w,x,y,z]
        
    Returns:
        body_vel: (3,) velocity in body frame
    """
    body_vel = np.zeros(3)
    quat_inv = np.zeros(4)
    mujoco.mju_negQuat(quat_inv, body_quat)
    mujoco.mju_rotVecQuat(body_vel, world_vel, quat_inv)
    return body_vel
```

### **main.py에서 사용**

```python
# motion_anchor_pos_b: 상대 변환 (이미 body frame)
anchor_pos_track_error, _ = compute_relative_transform_mujoco(...)
obs[...] = anchor_pos_track_error  #  No additional transform needed

# base_lin_vel: 벡터 회전 (명시적 변환)
world_lin_vel = mj_data.qvel[0:3]
robot_quat = mj_data.xquat[mujoco_anchor_body_id]
root_lin_vel = transform_velocity_to_body_frame(world_lin_vel, robot_quat)
obs[...] = root_lin_vel  #  Now in body frame
```

---

## 성능 개선 확인

### **수정 전**
```python
obs[...] = mj_data.qvel[0:3]  #  World frame (wrong!)
```
**결과**: SE policy 발산, 로봇 넘어짐

### **수정 후**
```python
obs[...] = transform_velocity_to_body_frame(mj_data.qvel[0:3], robot_quat)  #  Body frame (correct!)
```
**결과**: SE policy 성능 "무지하게 좋아짐"

---

##  결론

### **motion_anchor_pos_b를 바꾸지 않는 이유**
 **이미 올바름!** `compute_relative_transform_mujoco()`가 상대 변환을 통해 자동으로 body frame으로 변환

### **base_lin_vel을 바꾼 이유**
 **명시적 변환 필요!** World frame 속도를 body frame으로 회전 변환 필요

### **핵심 차이**
- **상대 변환** (T_A^(-1) * T_B): 좌표계 변환 자동 포함
- **벡터 회전** (R^T * v): 좌표계 변환 명시적 필요

---

**작성일**: 2025-11-01  
**작성자**: Coordinate Frame Analysis  
**상태**: 검증 완료 - 성능 개선 확인됨

