# EXECUTE_LOG

## Project Snapshot

- Current phase: `Phase 1`
- Primary track: Build the minimum 3-vehicle platoon environment, metrics, scenarios, and verification entrypoint
- Last updated: `2026-03-22`
- Completed modules:
  - `docs/problem_definition.md`
  - `docs/io_spec.md`
  - `docs/metrics_spec.md`
  - `envs/platoon_env.py`
  - `evaluation/platoon_metrics.py`
  - `scenarios/hazard_scenarios.py`
  - `scripts/verify_phase1.py`
- Current code baseline:
  - single-vehicle expert collection in `metadrive/exp_dataset/collect_expert.py`
  - single-vehicle diffusion training in `metadrive/policy/diffusion_policy/train_transfuser.py`
  - single-vehicle closed-loop testing in `metadrive/policy/diffusion_policy/test_transfuser_policy.py`

## Current Status

### Completed

- Phase 0 baseline documents created
- Execution log mechanism created
- Phase 0 completion state synchronized back to `AGENTS.md`
- Phase 1 environment skeleton created with dual action modes: `(8, 3)` trajectory and `(2,)` low-level control
- Phase 1 platoon metrics module created with 5 core episode-level metrics
- Phase 1 hazard scenario registry created with 3 minimum named scenarios
- Phase 1 task `1.1` acceptance passed on the `meta_drive` environment
- Phase 1 task `1.2` acceptance passed on the `meta_drive` environment
- Phase 1 task `1.3` acceptance passed on the `meta_drive` environment
- Phase 1 task `1.4` acceptance passed on the `meta_drive` environment
- Phase 1 task `1.5` rollout acceptance passed on the `meta_drive` environment
- Phase 1 task `1.5` rollout acceptance re-validated on the default `SSXCOCSS` map with background traffic
- Phase 1 task `1.6` info field enrichment acceptance passed on the `meta_drive` environment
- Phase 1 task `1.7` hazard scenario integration acceptance passed on the `meta_drive` environment

### In Progress

- Preparing handoff from completed `Phase 1` to the next phase

### Blockers

- No active blockers for `Phase 1`

### Next Step

- Start `Phase 2` data collection work or consume `Phase 1` outputs in `Phase 5/6`

## Phase Checklist

- `Phase 0`: `completed`
- `Phase 1`: `completed`
- `Phase 2`: `not_started`
- `Phase 3`: `not_started`
- `Phase 4`: `not_started`
- `Phase 5`: `not_started`
- `Phase 6`: `not_started`
- `Phase 7`: `not_started`

## Task Log

### 2026-03-21 — Phase 0 baseline and execution log

- 修改目标:
  创建 Phase 0 三份规范文档，并新增 `EXECUTE_LOG.md` 作为上下文恢复与进度记录入口
- 涉及文件:
  - `docs/problem_definition.md`
  - `docs/io_spec.md`
  - `docs/metrics_spec.md`
  - `EXECUTE_LOG.md`
  - `AGENTS.md`
- 关键设计选择:
  - 严格按 `AGENTS.md` 顺序推进，先完成 Phase 0 再进入 Phase 1
  - `EXECUTE_LOG.md` 放在项目根目录，单独承担执行状态与交接职责
  - 文档内容与 `AGENTS.md` 当前冻结的 3 车、纵列、12 维 `formation_relation_state`、`(8, 3)` 轨迹输出保持一致
- 最小测试方式:
  - `pytest metadrive/tests/test_policy/test_phase0_docs.py -q`
- 风险 / 未决事项:
  - 编队环境与指标接口尚未实现
  - `Phase 1` 仍需要结合 `MultiAgentMetaDrive` 和 `base_multi_env.py` 做进一步接口细化

### 2026-03-21 — Phase 1 platoon environment skeleton

- 修改目标:
  落地 `AGENTS.md` 的 Phase 1 最小骨架，包括 3 车编队环境、5 项指标、危险场景注册表和 IDM 验证脚本，并保持日志同步
- 涉及文件:
  - `envs/__init__.py`
  - `envs/platoon_env.py`
  - `evaluation/__init__.py`
  - `evaluation/platoon_metrics.py`
  - `scenarios/__init__.py`
  - `scenarios/hazard_scenarios.py`
  - `scripts/verify_phase1.py`
  - `metadrive/tests/test_policy/test_phase1_platoon_interfaces.py`
  - `AGENTS.md`
- 关键设计选择:
  - 环境主接口固定为每车 `(8, 3)` 轨迹，同时增加 `(2,)` 低层控制兼容模式，兼容 IDM 验证与调试
  - `formation_relation_state` 固定为 12 维，按其余编队成员的稳定顺序写入 `Δx, Δy, Δheading, Δv, Δx_target, Δy_target`
  - 指标模块只聚焦 Phase 1 所需的 5 项 episode 级指标，不提前扩展训练期复杂统计
  - 为了在轻量测试环境下保持接口可验证，`PlatoonEnv` 增加了 `stub_mode` 回退路径
- 最小测试方式:
  - `pytest metadrive/tests/test_policy/test_phase1_platoon_interfaces.py -q`
  - `python -m py_compile envs/platoon_env.py evaluation/platoon_metrics.py scenarios/hazard_scenarios.py scripts/verify_phase1.py`
  - `python scripts/verify_phase1.py --help`
  - `python scripts/verify_phase1.py --episodes 1 --render 0`
  - `python scripts/verify_phase1.py --episodes 5 --render 0`
- 风险 / 未决事项:
  - `1.5` 仍未完成，需继续推进真实 rollout 验收
  - 危险工况目前是“注册表 + 配置骨架”，不是完整高保真注入器

### 2026-03-22 — Phase 1 tasks 1.1~1.3 acceptance refresh

- 修改目标:
  按 `docs/phases/phase1.md` 的新版验收口径，真实完成并验收 `1.1~1.3`
- 涉及文件:
  - `envs/platoon_env.py`
  - `tests/acceptance/test_phase1_task1.py`
  - `tests/acceptance/test_phase1_task2.py`
  - `tests/acceptance/test_phase1_task3.py`
- 关键设计选择:
  - `PlatoonEnv` 保持继承 `BaseMultiEnv`，并继续支持 `(8,3)` 轨迹模式与 `(2,)` 低层模式
  - 初始间距默认车长改为与 `BaseMultiEnv` 默认 `xl` 车型一致的 `5.74m`，使真实 reset 后的间距满足 `vehicle_length + headway_time * speed`
  - `1.2` 和 `1.3` 不扩大范围，只按 Phase 文档规定的指标与场景注册结构验收
- 最小测试方式:
  - `/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase1_task1.py -v`
  - `/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase1_task2.py -v`
  - `/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase1_task3.py -v`
- 风险 / 未决事项:
  - `1.5~1.7` 仍未完成
  - 当前日志尚未记录 `verify_phase1.py` 的 5 episode 集成验收结果，因为那属于 `1.5`

### 2026-03-22 — Phase 1 tasks 1.4~1.5 completion

- 修改目标:
  完成 `verify_phase1.py` 验收脚本与 Phase 1 集成 rollout 验收，并按 `docs/phases/phase1.md` 的硬性指标真实通过
- 涉及文件:
  - `scripts/verify_phase1.py`
  - `envs/platoon_env.py`
  - `tests/acceptance/test_phase1_task4.py`
  - `tests/acceptance/test_phase1_task5.py`
  - `logs/phase1_acceptance.log`
  - `docs/phases/phase1.md`
  - `AGENTS.md`
- 关键设计选择:
  - `verify_phase1.py` 顶层显式 import `PlatoonEnv` 与 `IDMPolicy`，满足 `1.4` 的接口约束
  - 验收 rollout 使用短直道 `hybrid_map_sequence='S'`、`traffic_density=0.0`、禁用 IDM 自动换道
  - 对 IDM 跟驰参数进行编队化设置：`TIME_WANTED=0.1`、`DISTANCE_WANTED=1.0`，使车间距稳定维持在目标附近
  - 环境默认关闭压线即失败：`on_continuous_line_done=False`、`on_broken_line_done=False`，只在真正驶离道路时判失败
  - rollout summary 使用逐 episode 的真实运行统计，不伪造或手写指标
- 最小测试方式:
  - `/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase1_task4.py -v`
  - `/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase1_task5.py -v`
  - `timeout 60s /home/kong/anaconda3/envs/meta_drive/bin/python scripts/verify_phase1.py --episodes 5 --render 0 2>&1 | tee logs/phase1_acceptance.log`
- 风险 / 未决事项:
  - `Phase 1` 还剩 `1.6~1.7`
  - 当前 `success` 的 rollout 判定依赖“短直道验证场景中 3 车全部安全完成后被环境移除”，这是为 Phase 1 集成验收专门收口的判定

### 2026-03-22 — Phase 1 task 1.5 strict re-validation on default map

- 修改目标:
  按最新要求重做 `1.5`，禁止使用“短直道 + 无背景交通”配置，在默认 `SSXCOCSS` 地图和背景交通下完成真实 rollout 验收
- 涉及文件:
  - `envs/platoon_env.py`
  - `scripts/verify_phase1.py`
  - `logs/phase1_acceptance.log`
- 关键设计选择:
  - 保留 `enable_idm_lane_change=False`，不再覆盖 `hybrid_map_sequence` 与 `traffic_density`
  - 压线不再判失败，只有 `out_of_road=True` 才作为整队失败条件
  - 将 `formation_error` 改为基于相邻编队车中心距与目标间距的误差，避免弯道路段把正常同车道跟驰误计为大横向偏差
  - 重新整定 IDM：`TIME_WANTED=0.05`、`DISTANCE_WANTED=0.5`、`target_speed=18km/h`、`ACC_FACTOR=0.8`、`DEACC_FACTOR=-5.0`
  - 当所有 agent 都已完成并从环境中移除时，`PlatoonEnv` 直接置 `terminated['__all__']=True`
- 最小测试方式:
  - `python -m py_compile envs/platoon_env.py scripts/verify_phase1.py`
  - `/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase1_task5.py -v`
  - `/home/kong/anaconda3/envs/meta_drive/bin/python scripts/verify_phase1.py --episodes 5 --render 0`
- 实际结果:
  - `episode=1 success=1 collision=0 formation_error=1.832 min_gap=8.852`
  - `episode=2 success=1 collision=0 formation_error=1.831 min_gap=8.852`
  - `episode=3 success=1 collision=0 formation_error=1.832 min_gap=8.852`
  - `episode=4 success=1 collision=0 formation_error=1.832 min_gap=8.852`
  - `episode=5 success=1 collision=0 formation_error=1.832 min_gap=8.852`
  - `summary: success_rate=1.000 collision_rate=0.000 formation_error=1.832 recovery_time=0.000 min_inter_vehicle_gap=8.852`
- 风险 / 未决事项:
  - `phase1.md` 中旧的 `1.5` 实际结果仍是此前短直道配置下的日志；根据要求本次未改文档，只在执行日志中记录新的严格验收结果
  - `1.6~1.7` 仍待完成

### 2026-03-22 — Phase 1 tasks 1.6~1.7 completion

- 修改目标:
  完成 `1.6` 的奖励函数对接字段补充，以及 `1.7` 的 `hazard_scenario` 到 `PlatoonEnv` 的真实集成，并按 `phase1.md` 要求完成验收
- 涉及文件:
  - `envs/platoon_env.py`
  - `scenarios/hazard_scenarios.py`
  - `tests/acceptance/test_phase1_task6.py`
  - `tests/acceptance/test_phase1_task7.py`
  - `AGENTS.md`
  - `EXECUTE_LOG.md`
- 关键设计选择:
  - `info` 中新增并真实计算 `progress`、`jerk`、`delta_steering`、`speed_km_h`
  - `progress` 采用“同 lane 用纵向坐标差、跨 lane 用平面位移兜底”的方式，保证 3 步 IDM rollout 后为正
  - `jerk` 与 `delta_steering` 基于连续两步低层控制动作差分得到，确保为有限 float
  - `hazard_scenario` 在环境初始化前解析并应用 `env_overrides`，不再把该字段错误透传给 MetaDrive 原生配置
  - `static_obstacle_detour` 的 `traffic_density` 按 `phase1.md` 验收要求固定为 `0.15`
- 最小测试方式:
  - `python -m py_compile envs/platoon_env.py scenarios/hazard_scenarios.py tests/acceptance/test_phase1_task6.py tests/acceptance/test_phase1_task7.py`
  - `/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase1_task6.py -v`
  - `/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase1_task7.py -v`
  - `/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase1_task3.py -v`
  - `/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase1_task5.py -v`
- 风险 / 未决事项:
  - `Phase 1` 现已完成，后续若进入 `Phase 5` 还需要继续验证这些 `info` 字段是否满足奖励数值稳定性，而不仅是字段存在性
  - `EXECUTE_LOG.md` 中保留了 `1.5` 的两次不同验收背景记录，后续引用结果时应优先使用默认地图+背景交通的那次

## Open Decisions

- `PlatoonEnv` 的真实 MetaDrive 后端依赖需要如何在当前环境中稳定导入并运行
- Phase 1 完成后，危险工况是继续增强注入逻辑，还是直接转入 Phase 2 数据统计

## Handoff Notes

- 如果上下文不足，先读本文件，再读：
  1. `AGENTS.md`
  2. `docs/problem_definition.md`
  3. `docs/io_spec.md`
  4. `docs/metrics_spec.md`
- 当前已完成 Phase 1 的 `1.1-1.7`，并已在 `meta_drive` 解释器下真实跑过 acceptance
- 下一个最优先动作是切到 `Phase 2`，或直接消费 `Phase 1` 的 `info` 字段与 hazard 场景接口推进 `Phase 5/6`
- 执行 Phase 1 后续修复时，必须继续同步更新：
  - `AGENTS.md` 的完成标记
  - `EXECUTE_LOG.md` 的 `Current Status` 和 `Task Log`
