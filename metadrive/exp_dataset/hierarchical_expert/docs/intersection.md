交叉口通行速度调节方案

目标
在 ego 接近或进入交叉口时，不再只依赖 IDM 跟车，而是基于上帝视角背景车状态识别潜在冲突车，并通过调节纵向加速度主动避让，降低交叉口碰撞风险。

核心结构
新增一个规则类：

IntersectionConflictRegulator
输入：

ego 状态：position / speed / heading / lane / navigation
背景车列表
ego 当前参考路径或当前参考车道
IDM 原始输出加速度 a_idm
输出：

交叉口冲突调节后的纵向加速度 a_final
整体流程

判断 ego 是否位于交叉口附近
只有在接近交叉口时才启用该规则，普通路段继续使用 IDM。

筛选候选背景车
从环境中读取背景车状态，只保留：

距 ego 或交叉口中心较近的车辆
速度非零、仍在运动中的车辆
预测 ego 与候选车的未来运动

ego：沿当前参考车道/参考轨迹做短时预测
他车：用常速度模型预测短时轨迹
预测时域建议：

2~4s
时间步长 0.2s
识别冲突车
对每辆候选车，判断其未来轨迹是否与 ego 未来轨迹在交叉口区域内发生时空重叠。
可用两个指标：
最小空间距离
到达冲突点/冲突区的时间差
计算消解冲突所需加速度
对最危险冲突车，计算 ego 为“延后进入冲突区”所需的加速度。
设：

s_ego：ego 到冲突点的距离
v_ego：ego 当前速度
s_obj：对方到冲突点的距离
v_obj：对方当前速度
t_obj = s_obj / max(v_obj, eps)
Delta_safe：安全时间间隔，如 1.5s
要求 ego 晚于对方通过冲突点：

t_ego >= t_obj + Delta_safe
由匀加速度公式：

s_ego = v_ego * t + 0.5 * a * t^2
可反解所需加速度：

a_conf = 2 * (s_ego - v_ego * t_target) / t_target^2
其中：

t_target = t_obj + Delta_safe
与 IDM 融合
最终纵向控制取更保守者：
a_final = min(a_idm, a_conf)
含义：

如果 IDM 想加速，但交叉口冲突要求减速，则以冲突规则为准
如果 IDM 本身已经在减速，则保留更保守结果
建议规则

无冲突：直接使用 a_idm
轻度冲突：缓慢减速
中度冲突：明显减速让行
高风险冲突：强制制动或接近停车
实现建议
优先做第一版最小闭环：

不判断复杂路权
不做抢行，只做减速让行
不替换 IDM，只在纵向控制前增加 conflict regulator
在 HierarchicalExpertIDMPolicy.act() 中，先得到 a_idm，再经 IntersectionConflictRegulator 输出 a_final
优点

结构简单，容易接入现有 expert policy
不影响现有换道规划与轨迹跟踪
比纯 IDM 更适合交叉口多方向来车场景
具备明确物理意义，可解释、可调试