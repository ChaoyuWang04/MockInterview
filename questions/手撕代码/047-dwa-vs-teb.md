---
difficulty: 简单
topic: 机器人规划/动态窗口法
summary: 解释 DWA 的速度窗口、轨迹滚动预测、避障评分和工程边界
tags: [面经, 待校对, 手撕代码, DWA, 路径规划, 机器人]
company: VIVO
mastered: false
highfreq: false
---

## 题目

动态窗口法（DWA）用于移动机器人局部路径规划和实时避障。解释它的速度空间搜索、动力学约束、轨迹评价和参数调优，并与 TEB 比较实时性、安全性和路径质量，说明两者各自的优势与局限。

## 要点

- 搜索变量是线速度 $v$ 与角速度 $\omega$，动态窗口由速度上限和当前时刻可达加速度共同裁剪。
- 对每个候选控制量向前滚动预测轨迹，先拒绝碰撞或刹停距离不足的轨迹，再评分。
- 评价函数通常组合目标朝向、对全局路径的偏离、障碍 clearance、速度与平滑性；各项需归一化。
- DWA 是滚动局部优化，计算快但不保证全局最优，易在 U 形障碍、狭窄通道和拥挤动态场景陷入局部问题。
- 所有预测时长、采样数、控制周期、速度与延迟阈值都依赖机器人和硬件，不能写成通用固定数字。

## 答案

在一个控制周期 $\Delta t$ 内，可达速度窗口是

$$
v\in[\max(v_{min},v_t-a_v\Delta t),\ \min(v_{max},v_t+a_v\Delta t)],
$$

角速度同理。每个候选 $(v,\omega)$ 用运动学模型滚动若干步，检查机器人轮廓与障碍的距离，再评分。核心伪代码如下：

```text
def dwa_step(state, goal, obstacles, cfg):
    window = reachable_velocity_window(
        state.v, state.omega,
        cfg.speed_limits, cfg.accel_limits, cfg.control_dt,
    )

    feasible = []
    for v in sample(window.v_min, window.v_max, cfg.num_v_samples):
        for omega in sample(window.w_min, window.w_max, cfg.num_w_samples):
            trajectory = rollout(
                state, v, omega,
                horizon=cfg.horizon,
                dt=cfg.rollout_dt,
            )
            clearance = min_clearance(
                trajectory,
                predict_obstacles(obstacles, cfg.horizon),
                cfg.robot_footprint,
            )
            if clearance <= 0:
                continue
            if stopping_distance(v, cfg.max_decel) >= clearance:
                continue

            features = {
                "goal": normalized_goal_progress(trajectory, goal),
                "path": normalized_path_alignment(trajectory, cfg.global_path),
                "clearance": normalized_clearance(clearance, cfg),
                "speed": normalized_speed(v, cfg),
                "smooth": normalized_control_change(state, v, omega, cfg),
            }
            score = sum(cfg.weights[name] * value
                        for name, value in features.items())
            feasible.append((score, v, omega, trajectory))

    if not feasible:
        return emergency_stop_command()
    return max(feasible, key=lambda item: item[0])[1:3]
```

实时性来自低维控制空间、有限预测窗和可并行候选评估。动态障碍需要预测其随时间的位置；若只把最新传感器点云当静态障碍，系统只是频繁重规划，并未真正建模运动。

粗略计算量与速度样本数、角速度样本数、轨迹离散步数、每步碰撞检查成本相乘。调参前应统一各评分项量纲，再在仿真与实机上联合调权；增大 clearance 权重通常更保守，增大速度权重更激进，但结论受归一化定义影响。

### 与 TEB 的比较

| 维度 | DWA | TEB |
|---|---|---|
| 搜索对象 | 在动态窗口内采样 $(v,\omega)$，向前滚动短轨迹 | 联合优化一串位姿及相邻位姿的时间间隔 |
| 计算特征 | 候选数和预测步数固定时，单周期计算量较容易控制 | 需要迭代求解带运动学、障碍和时间项的图优化问题，耗时更依赖初值与收敛 |
| 路径质量 | 短视窗和离散采样可能带来摆动、绕行或局部最优 | 能在一条轨迹上联合优化平滑性与时间，通常更容易得到平滑轨迹 |
| 安全机制 | 碰撞检测、clearance 和刹停可行性 | 障碍距离约束/代价、运动学约束和时间参数化 |
| 常见局限 | U 形障碍、狭窄通道、速度采样分辨率 | 狭窄或拓扑复杂场景可能对初值敏感，迭代开销和参数耦合更强 |

两者都不天然“更安全”。安全性取决于障碍预测、机器人轮廓、约束是否为硬约束、控制延迟和失败时的停车策略。DWA 常用于计算预算紧、控制周期固定的场景；TEB 更适合愿意支付优化开销来换取轨迹平滑和显式时间参数化的场景。实际系统还要和全局路径、感知更新频率及底层控制器一起验证。

## 知识点

- 速度空间、动态可达窗口、滚动预测、刹停可行性、局部规划、评价函数、动态障碍预测、TEB、图优化。
- 面经原题：[B006-G01-Q145](../../docs/references/面经原题.md#b006-g01-q145)、[B006-G01-Q171](../../docs/references/面经原题.md#b006-g01-q171)。
- 老师答案参考：[P009-Q145](../../docs/references/平台题/P009-LC-081-160.md#p009-q145)、[P009-Q171](../../docs/references/平台题/P009-LC-161-241.md#p009-q171)。

## 追问

以下均为平台页面追问，不计入面经原题：

- DWA 与 TEB、MPC 各有什么优缺点？
- 狭窄通道和密集人群中为什么容易失败，怎样改进？
- 怎样把 DWA 与 A*、RRT 等全局规划器结合？

## Note
