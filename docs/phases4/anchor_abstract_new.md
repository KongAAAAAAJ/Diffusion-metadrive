## 单车语义行为 Anchor 提取可执行计划
1. 目标与总体原则
1.1 目标

从单车 expert 数据集中提取一组可解释、可组合、可复用的轨迹 anchors，作为后续编队任务中单个 agent 的行为基元。

这些 anchors 不是直接描述编队行为，而是描述：

单车在不同语义行为下的典型轨迹原型
后续在编队控制/强化学习中，可由多个单车 anchor 组合成联合行为
1.2 核心原则

anchor 提取不再采用“对全量单车轨迹直接做全局 K-Means 聚类”的方式，而采用以下三阶段流程：

先按单车语义行为分桶
再在每个行为桶内按轨迹形态与风格强度细分
最后用真实 medoid 轨迹，而不是均值中心，作为 anchor
1.3 为什么这样做

这样做的原因是：

全局几何聚类容易把不同驾驶意图混到一起
单车 anchor 最终要用于编队中的组合，必须保证语义清晰
强化学习更需要“稳定的行为基元”，而不是“几何均值轨迹”
2. Anchor 提取的三层结构

anchor 的层次定义如下：

第一层：主语义行为类别

这是最重要的一层，用来保证 anchor 的“意图清晰”。

推荐的 7 个主语义类如下：

KEEP_LANE
FOLLOW_BRAKE
YIELD
LANE_CHANGE_LEFT
LANE_CHANGE_RIGHT
TURN_LEFT
TURN_RIGHT
第二层：轨迹形态细分

在同一个主语义类内，再根据轨迹几何和速度结构差异进行细分，提取若干子 anchor。

例如：

KEEP_LANE 内可细分为稳态巡航、轻微左偏巡航、轻微右偏巡航
FOLLOW_BRAKE 内可细分为轻度跟驰、强减速跟驰、低速跟车
LANE_CHANGE_LEFT 内可细分为保守左换道、正常左换道、激进左换道
第三层：风格强度差异

风格强度不是第一层主语义，而是附着在第二层轨迹形态上的“细粒度差异”。

例如：

conservative
normal
aggressive

这层不必单独做成标签体系，可以通过聚类自动体现。

3. 单车语义行为的定义细节

下面明确 7 个主语义类的定义与划分依据。

3.1 KEEP_LANE
含义

ego 沿当前 lane 或当前 route 的当前分支正常前进，不存在显著减速原因，不发生 lane 拓扑切换。

典型特征
末端横向位移小
航向变化小
没有进入相邻车道
没有显著减速
不存在明显让行行为
备注

KEEP_LANE 不包含：

跟驰减速
让行等待
左转右转
换道
3.2 FOLLOW_BRAKE
含义

ego 主要因前向同向车辆约束而减速、跟驰或低速保持。

典型特征
轨迹几何基本仍沿当前 lane
速度 profile 明显下降或低速保持
前车距离较近
减速原因来自同向前车，而非横向冲突区
备注

这类和 KEEP_LANE 的区别，不在几何，而在速度结构与约束来源。

3.3 YIELD
含义

ego 因通行权不足或横向冲突对象存在而减速、等待、creep 或停车。

典型特征
轨迹几何通常仍沿当前 lane
可能提前减速、短暂停车、低速探头
减速原因不是同向前车，而是：
交叉口冲突车
merge 主路车
fork / split 冲突对象
stop / give-way 场景
备注

YIELD 必须与 FOLLOW_BRAKE 分开，因为二者的决策语义不同。

3.4 LANE_CHANGE_LEFT
含义

ego 从当前 lane 进入左邻车道，且最终稳定在目标左侧 lane。

典型特征
末端横向位移接近一个 lane width
轨迹横向偏移单调或近单调增加
最终车道拓扑改变为左邻 lane
非路口左转
备注

这里指的是标准左换道，不包含左转通过交叉口。

3.5 LANE_CHANGE_RIGHT

定义与 LANE_CHANGE_LEFT 对称。

3.6 TURN_LEFT
含义

ego 通过交叉口或连接段进入左向目标 road branch，完成左转拓扑切换。

典型特征
明显航向变化
路由拓扑发生切换
曲率显著大于普通 keep-lane
通常伴随低速过弯
备注

左转不应并入 KEEP_LANE。
左转和左换道的本质区别在于：前者是路网拓扑切换，后者是相邻车道切换。

3.7 TURN_RIGHT

定义与 TURN_LEFT 对称。

4. 单车语义自动打标签的执行规则

这一部分是 anchor 提取前的数据预处理环节。目标是给每条单车轨迹样本分配主语义行为标签。

建议按以下顺序进行标签判定，保证每条样本最终只有一个主语义标签。

4.1 判定优先级

建议优先级如下：

TURN_LEFT / TURN_RIGHT
LANE_CHANGE_LEFT / LANE_CHANGE_RIGHT
YIELD
FOLLOW_BRAKE
KEEP_LANE
4.2 TURN_LEFT / TURN_RIGHT 判定

主要依据：

route branch 是否发生左/右转拓扑切换
末端 heading 相对起始 heading 变化是否超过阈值
最终位置是否进入左/右目标路段

不建议只依赖“轨迹弯曲程度”，应结合拓扑和 heading 变化共同判断。

4.3 LANE_CHANGE_LEFT / RIGHT 判定

主要依据：

末端所在 lane 是否变为当前 lane 的左/右邻 lane
末端横向位移是否接近 lane width
轨迹横向偏移是否持续向左/右推进
不处于左/右转拓扑切换
4.4 YIELD 判定

主要依据：

样本处于交叉口、merge、fork、split 或其他存在横向冲突的场景区域
无同向前车强约束，仍然出现显著减速、停车或 creep
冲突区内或冲突区入口有潜在优先车辆
4.5 FOLLOW_BRAKE 判定

主要依据：

当前 lane 前方存在同向车辆
速度明显下降、低速保持或制动
减速主因来自前车约束
不满足 yield 判定
4.6 KEEP_LANE 判定

若以上都不满足，则归为 KEEP_LANE。

5. 各语义桶内的轨迹细分方法

完成第一层语义分桶后，再在每个桶内做细分。

这一步不再直接用全轨迹 flatten 向量，而是构造更稳定的特征。

5.1 KEEP_LANE 桶内细分

重点区分：

稳态巡航
轻微左偏巡航
轻微右偏巡航

建议特征：

终点纵向位移
平均速度
末速度
平均横向偏移
横向偏移标准差
最大横向偏移
5.2 FOLLOW_BRAKE 桶内细分

重点区分：

轻度跟驰
强减速跟驰
低速排队跟车

建议特征：

平均速度
末速度
最大减速度
最小 TTC_front
前车最小距离
纵向位移长度
5.3 YIELD 桶内细分

重点区分：

提前减速让行
creep 型让行
停车等待型让行

建议特征：

最小速度
停车持续时间
creep 段长度
最大减速度
冲突区停留时间
5.4 LANE_CHANGE_LEFT / RIGHT 桶内细分

重点区分：

保守换道
正常换道
激进换道

建议特征：

换道总时长
最大横向速度
最大横向加速度
最大曲率
换道期间的速度变化
轨迹完成度（是否完整进入目标 lane）
5.5 TURN_LEFT / RIGHT 桶内细分

重点区分：

保守低速转弯
正常转弯
较激进转弯

建议特征：

弯道中最小速度
平均曲率
heading 变化总量
转弯持续时间
出弯后速度恢复情况
6. 聚类方式与 medoid 选择策略
6.1 聚类方式

建议采用：

规则分桶 + 桶内聚类
桶内优先用 K-Means 或 K-Medoids
不再做全局聚类
6.2 每个桶的 anchor 数量

不做一刀切固定分配，而是根据：

样本数量
类内离散程度
后续组合价值
来确定

推荐第一版总 anchor 数控制在 14～18 个。

一个推荐分配如下：

KEEP_LANE：3 个
FOLLOW_BRAKE：3 个
YIELD：2 个
LANE_CHANGE_LEFT：3 个
LANE_CHANGE_RIGHT：3 个
TURN_LEFT：2 个
TURN_RIGHT：2 个

总数 = 18

6.3 为什么不用 cluster center 直接做 anchor

因为 cluster center 通常是“均值轨迹”，可能不对应任何真实可执行轨迹。

因此每个聚类簇最终 anchor 应取：

簇中心最近的真实样本轨迹
即 medoid 轨迹

这样能保证：

轨迹真实存在
动力学更自然
后续作为扩散模型初始模板更稳定
7. Anchor 的最终输出形式

每个 anchor 除了保存轨迹本身，还应保存元信息，便于后续组合使用。

建议每个 anchor 记录以下字段：

anchor_id
behavior_mode
subtype_id
trajectory_xy
trajectory_yaw（如有）
speed_profile（如有）
aggressiveness_level
nominal_lane_delta
nominal_speed_type
source_sample_id

其中：

behavior_mode 表示主语义类
subtype_id 表示桶内细分类型
source_sample_id 表示该 medoid 来自哪个真实样本
8. Anchor 提取后的质量检查要求

在代码实现后，需要对提取出的 anchors 做质量验收。

8.1 语义纯度检查

检查每个 anchor 所在簇内的样本，是否语义一致。
例如：

一个 LANE_CHANGE_LEFT anchor 里不能混入大量 TURN_LEFT
一个 FOLLOW_BRAKE anchor 里不能混入大量 YIELD
8.2 几何可解释性检查

对每个 anchor 可视化：

BEV 上的轨迹
速度 profile
lane 拓扑关系

确保：

KEEP_LANE 轨迹不应大横移
LC_LEFT 轨迹应完整进入左 lane
TURN_LEFT 轨迹应呈现明显转弯
8.3 分布覆盖检查

统计 7 个主语义类在数据集中占比，以及被 anchor 覆盖的程度，避免：

大类过密，小类缺失
或者总数过多导致后续组合空间过大
8.4 Medoid 真实性检查

确保每个 anchor 确实对应真实样本，不是均值插值结果。

9. 推荐的第一版实施顺序

为了降低实现难度，建议分阶段实施。

第一阶段

先完成：

7 个主语义类的自动打标签
每类提取简单统计特征
每类先固定提取 1～3 个 medoid anchors
第二阶段

再细化：

针对 FOLLOW_BRAKE、YIELD、LC_LEFT/RIGHT 增加更稳定的桶内特征
调整每类 anchor 数量
第三阶段

再做：

基于后续强化学习效果反向评估哪些类需要增减 anchor
对部分细分类做合并或拆分

# 总体要求：
1. 先分桶，再聚类，不允许直接对全量轨迹全局 K-Means
2. 主语义行为必须按 7 类定义
3. 每类使用独立特征空间，不要求所有类共用同一特征
4. 最终 anchor 必须是 medoid 真实轨迹，不是均值中心
5. 最终输出必须保留 anchor 的元标签
6. 提供可视化与统计接口，用于人工验收