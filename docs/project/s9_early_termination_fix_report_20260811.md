# S9 异常终止修复与验证报告（2026-08-11）

## 结论

S9 在第 25 帧附近无碰撞提前终止的问题已修复。最终代码使用固定种子
`17, 23, 31, 47, 59` 各运行 200 步，五集均达到 `200/200` 规划成功，
无碰撞、无出界、无 planner failure，并由 horizon 正常截断。

本报告只确认“异常提前终止”已消除。五集目前仍为
`scenario_functional_gate_failed`：三辆 ego 均已换到 lane 0，但尚未满足
“全部纵向超过 blocker”。因此 S9 完整功能验收和 v2 冻结仍未完成。

## 根因

1. 原候选排序主要惩罚空间曲率，没有显式考虑 `yaw_rate = speed × curvature`。
   换道持续时间相近的候选可能具有相近空间曲率，但闭环时域可跟踪性不同。
2. S9 连接段上的横向 cross-track 修正会与预瞄航向修正竞争，使无碰撞轨迹的
   航向误差越过既有 `0.1 rad` 硬阈值。
3. 串行绕行时，暂时 KEEP 的后车可能选中“刹停后恢复”轨迹。接近零速时，
   厘米级航向噪声会被转换为过大的空间曲率，导致 committed trajectory
   kinematic audit 失败。
4. 三车同步 LEFT 在部分随机参数下没有安全联合解；原 RuleMaker 没有向
   NormalPlanner 提供有序串行 LEFT 候选。

## 修改

- S9 LEFT 使用 `5.0, 5.5, 6.0, 6.5 s` 候选持续时间，并按最大偏航角速度增加
  trackability 排序代价。
- 扩充 S9 纵向加速度网格和候选池，保留更多“减速但持续滚动”的候选。
- 对 S9 LEFT 以及原子 LEFT 机动中的临时 KEEP 成员强制完整轨迹最小速度
  `1.0 m/s`；换道完成后的普通 KEEP 只使用软排序，不做硬过滤。
- RuleMaker 先提交三车联合 LEFT；若完整时域审计不可行，再提交按编队顺序的
  单车 LEFT + 其余 KEEP 候选。
- S9 PID 关闭 cross-track 叠加修正，仅保留轨迹预瞄航向闭环；其他场景继续使用
  原 `0.4` 增益。
- 候选 debug 新增最大偏航角速度、trackability 代价、最小速度和速度代价。

未修改统一硬安全契约：航向误差 `0.1 rad`、纵向误差 `1.0 m`、横向误差
`0.5 m`、背景车最小间距 `5 m`、编队最小间距 `7 m`，也未修改控制器接口。

## 五种子验证

| seed | frames | planning | collision | out of road | lane 0 change | passed blocker | episode failure |
|---:|---:|---:|---|---|---|---|---|
| 17 | 200 | 200/200 | false | false | true | false | scenario_functional_gate_failed |
| 23 | 200 | 200/200 | false | false | true | false | scenario_functional_gate_failed |
| 31 | 200 | 200/200 | false | false | true | false | scenario_functional_gate_failed |
| 47 | 200 | 200/200 | false | false | true | false | scenario_functional_gate_failed |
| 59 | 200 | 200/200 | false | false | true | false | scenario_functional_gate_failed |

回归测试：

```text
223 passed, 205 warnings in 9.31s
```

## 最终 seed 17 证据

根目录：

```text
/tmp/diffusion-metadrive-s5-s9-revision/outputs/s5_s9_candidate_revision_eval_20260811/s9_early_termination_fix_final_seed17/S9_narrow_channel_negotiation
```

| artifact | size | decode result | SHA256 |
|---|---:|---|---|
| `video/episode_0000.mp4` | 86,391 B | H.264, 800×800, 20 s, 200 frames | `604cbf2e289029b7b23879d31d8f8152a3f9fb2e7b604520008373f00812a9d9` |
| `semantic_bev_video/episode_0000.mp4` | 440,102 B | H.264, 776×256, 20 s, 200 frames | `cf74e6a55ee980fe6e631272133333db7da8726160e8616bd58cd28d163dfe0a` |
| `trajectories/episode_0000.npz` | 45,060 B | non-empty | `677581f39c22196536820862d468f100f008f0b15f6137b150e55f81ff554b9e` |
| `metrices/episode_0000/expert_episode.json` | 14,380 B | valid JSON | `6f199d5f7185f121be14c6eb8166ab61e350c88be6ea86fba5836d61576a92b5` |

输出目录独立于旧评估结果，未写入旧数据、checkpoint 或训练元信息。
