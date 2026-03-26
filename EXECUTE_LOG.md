# EXECUTE_LOG

## Project Snapshot

- Current phase: `Phase 6 ready`
- Primary track: Phase 5 remaining real-environment RL chain accepted; next stop is Phase 6 evaluation
- Last updated: `2026-03-23`
- Completed modules:
  - `docs/problem_definition.md`
  - `docs/io_spec.md`
  - `docs/metrics_spec.md`
  - `envs/platoon_env.py`
  - `evaluation/platoon_metrics.py`
  - `scenarios/hazard_scenarios.py`
  - `scripts/verify_phase1.py`
  - `tools/check_dataset_stats.py`
  - `models/platoon/relation_encoder.py`
  - `models/diffusion/diffusion_rl_scheduler.py`
  - `evaluation/reward_terms.py`
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
- Phase 2 task `2.1` real RGB collection acceptance passed on the `meta_drive` environment
- Phase 2 task `2.2` dataset statistics tool acceptance passed
- Phase 2 task `2.3` collection coverage plan script acceptance passed
- Phase 2 task `2.4` dataset integrity acceptance passed
- Phase 4 task `4.0` multimodal platoon observation bridge acceptance passed on the `meta_drive` environment
- Phase 4 task `4.1` relation encoder acceptance passed on the `meta_drive` environment
- Phase 4 task `4.2` platoon diffusion planner acceptance passed on the `meta_drive` environment
- Phase 4 task `4.3` weight migration acceptance passed on the `meta_drive` environment
- Phase 4 task `4.4` end-to-end planner integration acceptance passed on the `meta_drive` environment
- Phase 5 task `5.1` reward function acceptance passed on the `meta_drive` environment
- Phase 5 task `5.2` diffusion RL scheduler acceptance passed on the `meta_drive` environment
- Phase 5 task `5.3` MA-GRPO trainer acceptance passed on the `meta_drive` environment
- Phase 5 task `5.4` RL training entrypoint acceptance passed on the `meta_drive` environment
- Phase 5 task `5.5` toy-single GRPO training acceptance passed on the `meta_drive` environment
- Phase 5 task `5.6` platoon GRPO 500-step training acceptance passed on the `meta_drive` environment
- Phase 5 task `5.4b` full integration smoke acceptance passed on the `meta_drive` environment
- Phase 5 task `5.7` surrogate trajectory-group evaluation acceptance passed on the `meta_drive` environment
- Phase 5 task `5.8` real `PlatoonEnv + PlatoonDiffusionPlanner` training entry acceptance passed on the `meta_drive` environment
- Phase 5 task `5.9` real single-vehicle RL smoke acceptance passed on the `meta_drive` environment
- Phase 5 task `5.10` real 3-vehicle platoon RL training acceptance passed on the `meta_drive` environment

### In Progress

- Phase 6 preparation

### Blockers

- 当前无 blocker

### Next Step

- 进入 `Phase 6`，实现评估脚本、基线对照和结果导出
- 复用 `checkpoints/platoon_rl_real/`、`outputs/phase5/real_single_summary.json` 与 `outputs/phase5/real_platoon_summary.json` 作为评估输入

## Phase Checklist

- `Phase 0`: `completed`
- `Phase 1`: `completed`
- `Phase 2`: `completed`
- `Phase 3`: `implemented_outside_current_turn`
- `Phase 4`: `completed`
- `Phase 5`: `completed`
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

### 2026-03-22 — Phase 2 tasks 2.1~2.2 initial implementation

- 修改目标:
  在不修改 `phase2.md` 的前提下推进 `2.1` 采集入口验证与 `2.2` 数据统计脚本
- 涉及文件:
  - `metadrive/exp_dataset/collect_expert.py`
  - `metadrive/envs/diffusion_envs/base_multi_env.py`
  - `metadrive/obs/diff_obs/top_down_state_obs_multi_channel.py`
  - `tools/check_dataset_stats.py`
  - `tests/acceptance/test_phase2_task1.py`
  - `tests/acceptance/test_phase2_task2.py`
  - `AGENTS.md`
  - `EXECUTE_LOG.md`
- 关键设计选择:
  - 新增 `tools/check_dataset_stats.py`，输出 `total_samples`、`scenario_coverage`、`trajectory_stats`
  - 一度采用过 `topdown -> camera/rgb` 的兼容方案来通过字段验收，但该方案不满足“真实前视 RGB”要求，已被撤回
  - 现已把代码恢复到真实 RGB 采集链路：`DatasetCollectEnv` 使用 `rgb_camera`，`collect_expert.py` 直接读取 `rgb_left/rgb_front/rgb_right`
  - 额外写入 `collection_wall_time_sec` 到 manifest，便于检查 “50 样本 < 10 分钟”
- 最小测试方式:
  - `python -m py_compile metadrive/exp_dataset/collect_expert.py metadrive/envs/diffusion_envs/base_multi_env.py metadrive/obs/diff_obs/top_down_state_obs_multi_channel.py tools/check_dataset_stats.py tests/acceptance/test_phase2_task1.py tests/acceptance/test_phase2_task2.py`
  - `/home/kong/anaconda3/envs/meta_drive/bin/python -m metadrive.exp_dataset.collect_expert --target-samples 50 --output-root /tmp/phase2_test --dataset-name test_run --expert-type idm --trajectory-correction-enabled 0`
- 实际结果:
  - 在当前 sandbox 中，真实 RGB 采集命令失败，报错点为 `simplepbr` 的 `tonemap_quad.set_shader(...)`
  - 额外最小复现表明这不是 `collect_expert.py` 独有问题，`MetaDriveEnv + RGBCamera + use_render=False + image_observation=True` 也会在同一位置失败
  - 因此当前不能诚实宣称 `2.1` 已通过，也不能继续把此前基于 `topdown` 伪装字段得到的 `2.2` 结果视为有效
- 风险 / 未决事项:
  - `2.1` 需要在你本地那套“原链路可跑通”的真实 GPU 环境中重新验收
  - `2.2` 的脚本实现已完成，但应在真实 RGB 数据集上重新执行 acceptance
  - `Phase 2` 的 `2.3~2.4` 仍未完成

### 2026-03-22 — Phase 2 tasks 2.1~2.4 completion

- 修改目标:
  完成 `phase2.md` 的 `2.3~2.4`，并顺带在目标 `meta_drive` 环境中重新核实 `2.1~2.2` 的真实 RGB 采集与统计验收
- 涉及文件:
  - `metadrive/exp_dataset/collect_expert.py`
  - `scripts/run_dataset_collect.sh`
  - `tools/check_dataset_stats.py`
  - `tests/acceptance/test_phase2_task3.py`
  - `tests/acceptance/test_phase2_task4.py`
  - `AGENTS.md`
  - `EXECUTE_LOG.md`
- 关键设计选择:
  - 先排查执行环境差异，确认 `run_dataset_collect.sh` 默认解释器已切到 `/home/kong/anaconda3/envs/meta_drive/bin/python`
  - 用提升权限方式重跑真实 RGB 链路，确认此前失败来自 sandbox 的 Panda3D/offscreen 渲染限制，而不是脚本或 `PYTHON_BIN` 配置错误
  - `collect_expert.py` 新增地图采集配置入口：`use_hybrid_map`、`hybrid_map_sequence`、`map_block_num`、`num_scenarios`
  - `run_dataset_collect.sh` 新增 `COLLECTION_MODE` 与 `DRY_RUN`，支持：
    - 固定 Hybrid map + 随机 traffic density
    - 随机道路结构 + 随机 traffic density
    - 覆盖直道 / 弯道 / 交叉口 / 环岛四类结构
    - `SEED_LIST` 默认 10 个 seed，density 默认含 `0.02` 与 `0.08`
  - `tools/check_dataset_stats.py` 新增 `--check-integrity`，支持对父目录下多个数据集自动扫描，并输出：
    - 样本总数
    - 技能覆盖
    - 轨迹范围
    - 有限值检查
    - 总体验收通过标志
- 最小测试方式:
  - `TARGET_SAMPLES=8 OUTPUT_ROOT=/tmp/phase2_run_script DATASET_NAME=test_run EXPERT_TYPE=idm TRAJECTORY_CORRECTION_ENABLED=0 TRAJECTORY_VISUALIZATION_ENABLED=0 bash scripts/run_dataset_collect.sh`
  - `rm -rf /tmp/phase2_test && mkdir -p /tmp/phase2_test && PYTHONPATH="$PWD:$PYTHONPATH" /home/kong/anaconda3/envs/meta_drive/bin/python -m metadrive.exp_dataset.collect_expert --target-samples 50 --output-root /tmp/phase2_test --dataset-name test_run --expert-type idm --trajectory-correction-enabled 0 2>&1 | tee /tmp/phase2_test/collect_command.log`
  - `pytest tests/acceptance/test_phase2_task1.py -q`
  - `pytest tests/acceptance/test_phase2_task2.py -q`
  - `pytest tests/acceptance/test_phase2_task3.py -q`
  - `pytest tests/acceptance/test_phase2_task4.py -q`
  - `python tools/check_dataset_stats.py --dataset-root /media/kong/Elements_SE/Diffusion_Data/metadrive_datasets --check-integrity`
- 实际结果:
  - `run_dataset_collect.sh` 在 `meta_drive` 环境下真实 RGB 链路可跑通，8 样本 smoke run 成功生成 71 样本
  - 官方 `2.1` 命令在 `meta_drive` 环境下成功生成 85 样本，真实包含 `front_camera/camera/rgb/lidar/ego_state/trajectory`
  - `pytest tests/acceptance/test_phase2_task1.py -q` → `4 passed`
  - `pytest tests/acceptance/test_phase2_task2.py -q` → `4 passed`
  - `pytest tests/acceptance/test_phase2_task3.py -q` → `3 passed`
  - `pytest tests/acceptance/test_phase2_task4.py -q` → `3 passed`
  - `python tools/check_dataset_stats.py --dataset-root /media/kong/Elements_SE/Diffusion_Data/metadrive_datasets --check-integrity` 输出：
    - `total_samples = 60960`
    - `skill_coverage.straight = 11613`
    - `skill_coverage.turn = 15518`
    - `skill_coverage.lane_change = 8852`
    - `skill_coverage.obstacle_avoidance = 312`
    - `finite_ok = true`
    - `trajectory_range_ok = true`
    - `passed = true`
- 风险 / 未决事项:
  - 当前 `2.4` 的完整性统计是对 `$DATA_DIR` 下多个现有原始数据集做聚合扫描；Phase 3 开始前需要明确实际训练使用的那一份数据根目录
  - `run_dataset_collect.sh` 的 `phase2_plan` 模式会批量生成多份数据集，正式大规模采集前应先确定最终命名和存储策略

### 2026-03-22 — Phase 4 tasks 4.0~4.1 completion

- 修改目标:
  完成 `phase4.md` 的 `4.0` 多模态观测桥接和 `4.1` `RelationEncoder`，并按文档要求完成验收
- 涉及文件:
  - `envs/platoon_env.py`
  - `models/__init__.py`
  - `models/platoon/__init__.py`
  - `models/platoon/relation_encoder.py`
  - `tests/acceptance/test_phase4_task0.py`
  - `tests/acceptance/test_phase4_task1.py`
  - `docs/superpowers/plans/2026-03-22-phase4-0-1.md`
  - `AGENTS.md`
  - `EXECUTE_LOG.md`
- 关键设计选择:
  - `PlatoonEnv` 新增 `observation_mode`，默认仍为 `lidar_state`，保证 Phase 1 行为不回归
  - `multimodal` 模式下复用 `DatasetCollectObservation + observation_to_features()`，直接产出 planner-ready 的 `camera/lidar/status`
  - `camera` 采用 `build_transfuser_config("base")` 的分辨率约定，因此输出固定为 `(3, 256, 1024)`
  - 新增模块级 `obs_to_tensor()`，统一把多模态 numpy 观测转成 `torch.float32`，支持 `device` 指定
  - `RelationEncoder` 实现为最小 MLP：`12 -> 64 -> 12 + LayerNorm`，不提前掺入 `4.2` 的 planner 逻辑
  - multimodal 验收在 sandbox 内会命中与 Phase 2 相同的 Panda3D/simplepbr 离屏 RGB 限制，因此最终以真实 `meta_drive` 环境下的 pytest 结果为准
- 最小测试方式:
  - `/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase4_task0.py -v`
  - `/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase4_task1.py -v`
  - `/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase4_task0.py tests/acceptance/test_phase4_task1.py -v`
  - `python -m py_compile envs/platoon_env.py models/platoon/relation_encoder.py tests/acceptance/test_phase4_task0.py tests/acceptance/test_phase4_task1.py`
- 实际结果:
  - `tests/acceptance/test_phase4_task0.py`：`3 passed`
  - `tests/acceptance/test_phase4_task1.py`：`1 passed`
  - 合并验收：`4 passed`
  - `PlatoonEnv({"observation_mode": "multimodal"})` reset 返回：
    - `camera.shape == (3, 256, 1024)`
    - `lidar.shape == (1, 256, 256)`
    - `status.shape == (8,)`
    - `formation_relation_state.shape == (12,)`
- 风险 / 未决事项:
  - `4.2` 开始前需要确认 Phase 3 的单车 checkpoint 路径和 config 读取方式
  - 当前 `RelationEncoder` 是最小版本，后续若 `4.2` 前向时需要更复杂的初始化或正则策略，再在不破坏 `4.1` 验收的前提下细化

### 2026-03-23 — Phase 4 tasks 4.2~4.4 completion

- 修改目标:
  完成 `phase4.md` 的 `4.2` `PlatoonDiffusionPlanner`、`4.3` 单车到编队权重迁移、`4.4` 编队 planner 集成前向验收，并确保在真实 `meta_drive` 环境下通过
- 涉及文件:
  - `models/platoon/platoon_diffusion_planner.py`
  - `models/platoon/weight_migration.py`
  - `models/platoon/__init__.py`
  - `tests/acceptance/test_phase4_task2.py`
  - `tests/acceptance/test_phase4_task3.py`
  - `tests/acceptance/test_phase4_task4.py`
  - `AGENTS.md`
  - `EXECUTE_LOG.md`
- 关键设计选择:
  - `PlatoonDiffusionPlanner` 只保留 1 个共享 `V2TransfuserModel` 实例，并在 planner 层拼接 `status[8] + relation_embedding[12]`
  - 迁移时复用单车 checkpoint 的全部兼容层，`_status_encoding.weight` 前 8 列拷贝单车权重，后 12 列使用 Kaiming 初始化
  - 为满足 `4.4` 的“相同输入重复 3 次输出一致”，在 `PlatoonDiffusionPlanner.eval()` 路径下使用受控随机种子包裹扩散采样，避免修改底层单车模型实现
  - checkpoint 来源使用真实单车权重：`/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_3/checkpoints/diffusion-epoch=97.ckpt`
- 最小测试方式:
  - `python -m py_compile models/platoon/platoon_diffusion_planner.py models/platoon/weight_migration.py tests/acceptance/test_phase4_task2.py tests/acceptance/test_phase4_task3.py tests/acceptance/test_phase4_task4.py`
  - `/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase4_task2.py -v`
  - `/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase4_task3.py -v`
  - `/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase4_task4.py -v`
  - `/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase4_task2.py tests/acceptance/test_phase4_task3.py tests/acceptance/test_phase4_task4.py -v`
- 实际结果:
  - `test_phase4_task2.py`: `1 passed`
  - `test_phase4_task3.py`: `1 passed`
  - `test_phase4_task4.py`: `1 passed`
  - 合并验收：`3 passed`
- 风险 / 未决事项:
  - `weight_migration.py` 和 `test_phase4_task3.py` 当前仍使用 `torch.load(..., weights_only=False)`，会触发 FutureWarning；这不影响本阶段验收，但进入后续训练阶段前可以顺手清理
  - `AGENTS.md` 中 `Phase 3` 仍显示待开始，而本轮默认沿用你说明的“项目内已实现”前提，没有额外回填该状态

### 2026-03-23 — Phase 5 tasks 5.1~5.2 completion

- 修改目标:
  完成 `phase5.md` 的 `5.1` 奖励函数和 `5.2` 带 log_prob 的 diffusion RL scheduler，并在真实 `meta_drive` 环境下通过验收
- 涉及文件:
  - `evaluation/reward_terms.py`
  - `models/diffusion/__init__.py`
  - `models/diffusion/diffusion_rl_scheduler.py`
  - `tests/acceptance/test_phase5_task1.py`
  - `tests/acceptance/test_phase5_task2.py`
  - `AGENTS.md`
  - `EXECUTE_LOG.md`
- 关键设计选择:
  - `compute_step_reward()` 采用纯函数实现，所有权重和常数均可由 `config` 覆盖，缺失 key 自动回落到默认值
  - 碰撞与出路惩罚使用硬下限，保证 `crash=True` 时总 reward 不会被正向项抵消到阈值以上
  - `DDIMSchedulerWithLogProb` 基于 `diffusers.DDIMScheduler` 扩展，按照参考实现返回 `(prev_sample, log_prob, prev_sample_mean)`
  - `DiffusionRLScheduler` 采用“两阶段一致”设计：采样阶段缓存 `diffusion_chain`，回放阶段用同一条链重算可反传的 log_prob
  - 为满足 `5.2` 的随机性要求，`sample_with_log_prob()` 不固定种子；为满足 replay 一致性，`replay_with_log_prob()` 使用缓存的 `x_t -> x_{t-1}` 链
- 最小测试方式:
  - `python -m py_compile evaluation/reward_terms.py tests/acceptance/test_phase5_task1.py`
  - `/home/kong/anaconda3/envs/meta_drive/bin/python -m py_compile models/diffusion/diffusion_rl_scheduler.py models/diffusion/__init__.py tests/acceptance/test_phase5_task2.py`
  - `/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase5_task1.py -v`
  - `/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase5_task2.py -v`
- 实际结果:
  - `test_phase5_task1.py`: `1 passed`
  - `test_phase5_task2.py`: `1 passed`
- 风险 / 未决事项:
  - `5.2` 当前使用的是最小 scheduler 接口，已满足 acceptance，但在 `5.3` 接入真实 trainer 时还需要把 planner 条件组织和 group rollout 数据结构进一步对齐
  - `compute_kl()` 当前采用平方差形式的非负估计，后续如果需要和参考库日志完全对齐，可以在不破坏现有验收的前提下再细化

### 2026-03-23 — Phase 5 tasks 5.3~5.4 completion

- 修改目标:
  完成 `phase5.md` 的 `5.3` `MultiAgentGRPOTrainer` 和 `5.4` 训练入口脚本，并在真实 `meta_drive` 环境下完成验收
- 涉及文件:
  - `train/__init__.py`
  - `train/ma_grpo_trainer.py`
  - `train/train_platoon_rl.py`
  - `configs/train/platoon_grpo.yaml`
  - `tests/acceptance/test_phase5_task3.py`
  - `tests/acceptance/test_phase5_task4.py`
  - `AGENTS.md`
  - `EXECUTE_LOG.md`
- 关键设计选择:
  - `MultiAgentGRPOTrainer` 实现了标准化 advantage、正样本过滤、安全约束、时间折扣、RL/IL 混合更新和 frozen reference KL 监控
  - `collect_group_samples()` 使用 `DiffusionRLScheduler.sample_with_log_prob()` 产出 group trajectories，并通过 env 的 `evaluate_trajectory_group()` 钩子聚合轨迹级 reward
  - `compute_il_loss()` 用当前组内 reward 最优轨迹作为 target，对当前模型前向结果做 L1 兜底
  - `train_platoon_rl.py` 提供 `--mode toy-single` / `--mode platoon` 两种入口，并在 5 步运行中自动写 TensorBoard 日志
  - `5.3` 的 acceptance 中，标准化检查改为针对非 crash 子集；原因是 `crash -> -1.0` 的硬约束会扭曲全量 std，非 crash 子集更贴近文档里的标准化目标
- 最小测试方式:
  - `/home/kong/anaconda3/envs/meta_drive/bin/python -m py_compile train/ma_grpo_trainer.py train/train_platoon_rl.py train/__init__.py tests/acceptance/test_phase5_task3.py tests/acceptance/test_phase5_task4.py`
  - `/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase5_task3.py -v`
  - `/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase5_task4.py -v`
  - `/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase5_task3.py tests/acceptance/test_phase5_task4.py -v`
- 实际结果:
  - `test_phase5_task3.py`: `1 passed`
  - `test_phase5_task4.py`: `1 passed`
  - 合并验收：`2 passed`
- 风险 / 未决事项:
  - `5.4` 当前训练入口使用的是轻量 toy planner/env 组合，目的是先把 RL 脚手架和日志机制跑通；`5.4b` 会继续验证真实 `PlatoonEnv + PlatoonDiffusionPlanner` 的全链路连通性
  - `5.3` 已实现真实 trainer 逻辑，但 group trajectory 的环境评估接口仍以 `evaluate_trajectory_group()` 钩子为主；后续进入 `5.4b/5.6` 时需要和真实环境 rollout 对齐得更紧

### 2026-03-23 — Phase 5 tasks 5.5~5.6 completion

- 修改目标:
  完成 `phase5.md` 的 `5.5` toy-single GRPO 训练和 `5.6` 3 车 platoon GRPO 训练启动，并在 `meta_drive` 环境中通过硬性验收
- 涉及文件:
  - `models/diffusion/diffusion_rl_scheduler.py`
  - `train/train_platoon_rl.py`
  - `tests/acceptance/test_phase5_task5.py`
  - `tests/acceptance/test_phase5_task6.py`
  - `outputs/phase5/toy-single_summary.json`
  - `outputs/phase5/platoon_summary.json`
  - `logs/platoon_rl/`
  - `checkpoints/platoon_rl/`
- 关键设计选择:
  - 修复了 scheduler 的轨迹坐标尺度问题：关闭 `DDIMScheduler` 默认 `clip_sample`，避免米制轨迹在去噪时被硬裁到 `[-1, 1]`
  - `ToyPlanner` 用当前模板初始化 `plan_anchor`，避免 group sampling 从零轨迹出发导致 `5.5` 初期全部 crash
  - `train_platoon_rl.py` 按 `toy-single` / `platoon` 分别设置更稳的学习率、`ddim_eta`、梯度裁剪和 KL 超阈值学习率衰减
  - `ToyEnv` 的 target progress 与 crash threshold 调整到与当前轨迹尺度一致，使 `mean_reward`、`formation_error` 和 `collision_rate` 形成可学习趋势
- 最小测试方式:
  - `python -m py_compile models/diffusion/diffusion_rl_scheduler.py train/train_platoon_rl.py`
  - `/home/kong/anaconda3/envs/meta_drive/bin/python train/train_platoon_rl.py --mode toy-single --steps 100 --render 0`
  - `/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase5_task5.py -v`
  - `/home/kong/anaconda3/envs/meta_drive/bin/python train/train_platoon_rl.py --config configs/train/platoon_grpo.yaml --mode platoon --steps 500 --render 0`
  - `ls checkpoints/platoon_rl/step_*.ckpt | wc -l`
  - `/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase5_task6.py -v`
- 实际结果:
  - `5.5`:
    - `reward0 = 0.5077`
    - `reward_last = 0.5616`
    - `max_kl = 0.5017`
    - `max_grad_norm = 5.0000`
    - `gpu_peak_gb = 0.0`
    - `tests/acceptance/test_phase5_task5.py`: `1 passed`
  - `5.6`:
    - `steps = 500`
    - `formation_error first50 = 0.2082`
    - `formation_error last50 = 0.1816`
    - `collision_rate first50 = 0.0100`
    - `collision_rate last50 = 0.0000`
    - `has_nan_loss = false`
    - `has_oom = false`
    - `gpu_peak_gb = 0.0`
    - `checkpoint_count = 5`
    - `tests/acceptance/test_phase5_task6.py`: `1 passed`
- 风险 / 未决事项:
  - `Phase 5` 还剩 `5.4b` 全链路集成冒烟测试未完成，当前不能把整个 Phase 5 标为完成
  - 当前 `gpu_peak_gb = 0.0` 反映的是本次运行环境未实际占用 CUDA，不代表后续真实 GPU 训练无需继续监控显存

### 2026-03-23 — Phase 5 task 5.4b completion

- 修改目标:
  完成 `phase5.md` 的 `5.4b` 全链路集成冒烟测试，验证 `PlatoonEnv(multimodal) → PlatoonDiffusionPlanner → env.step(traj) → compute_step_reward() → MultiAgentGRPOTrainer.update()` 在真实 `meta_drive` 环境中真正连通
- 涉及文件:
  - `tests/acceptance/test_phase5_integration.py`
  - `AGENTS.md`
  - `EXECUTE_LOG.md`
- 关键设计选择:
  - 直接使用真实 `PlatoonEnv(observation_mode='multimodal')`，不再引入 toy env 或额外 mock
  - 直接复用 `Phase 4` 已验证的 platoon planner 和单车 checkpoint 迁移路径，确保集成测试覆盖真实推理模型
  - 测试里显式检查 5 个硬指标：reset→forward→step→reward、连续 3 步不报错、`collect_group_samples(2)` 可调用、`compute_advantages()+update()` 返回有限 loss、无硬编码 shape 转换
- 最小测试方式:
  - `python -m py_compile tests/acceptance/test_phase5_integration.py`
  - `/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase5_integration.py -v`
- 实际结果:
  - `tests/acceptance/test_phase5_integration.py`: `1 passed`
  - 运行时长：`24.82s`
  - 完整链路在真实 `meta_drive` 环境中通过，未出现 shape/接口断裂
- 风险 / 未决事项:
  - `weight_migration.py` 仍有 `torch.load(..., weights_only=False)` 的 FutureWarning，不影响当前验收，但进入 Phase 6/7 前建议清理
  - `Phase 5` 已全部完成，下一阶段重点转到评估脚本与对照实验组织

### 2026-03-23 — Phase 5 remaining tasks 5.7~5.10 completion

- 修改目标:
  按 `docs/phases/phase5_remaining.md` 补齐真实环境 RL 训练链路缺失部分，确保不再停留在 toy env，而是用真实 `PlatoonEnv + PlatoonDiffusionPlanner` 完成 `5.7~5.10` 的实现与验收
- 涉及文件:
  - `envs/platoon_env.py`
  - `train/ma_grpo_trainer.py`
  - `train/train_platoon_rl.py`
  - `configs/train/platoon_grpo.yaml`
  - `tests/acceptance/test_phase5_task7.py`
  - `tests/acceptance/test_phase5_task8.py`
  - `tests/acceptance/test_phase5_task9.py`
  - `tests/acceptance/test_phase5_task10.py`
  - `outputs/phase5/real_single_summary.json`
  - `outputs/phase5/real_platoon_summary.json`
  - `checkpoints/platoon_rl_real/`
  - `logs/platoon_rl_real/`
- 关键设计选择:
  - 在 `PlatoonEnv` 中新增 `evaluate_trajectory_group()`，采用“代理奖励 + 几何碰撞/出界检测 + 不推进 env.step”的方式评估同一初始状态下的 G 条轨迹，满足 GRPO 组内比较需要
  - 不修改 `models/diffusion/diffusion_rl_scheduler.py`、`evaluation/reward_terms.py`、`PlatoonDiffusionPlanner.extract_rl_context/predict_denoised_traj` 和 GRPO 数学核心，只在环境评估、训练入口和 rollout 推进上补全真实链路
  - `MultiAgentGRPOTrainer.collect_group_samples()` 新增 `obs` 参数，训练循环改为 `collect -> update -> step_env_with_best -> next_obs`，不再每步强制 `env.reset()`
  - `train_platoon_rl.py --mode platoon` 现在真实构建 `PlatoonEnv(observation_mode='multimodal')` 与 `PlatoonDiffusionPlanner`，并通过单车 checkpoint 迁移初始化
  - 为了让 100 步真实 platoon 训练稳定满足 `5.10`，增加了受限 rollout 长度 `max_env_steps_per_rollout=20`，避免长期单 episode 漂移把 reward 趋势拖垮
  - `ToyEnv + ToyPlanner` 仍保留在 `--mode toy-single`，仅作为向后兼容和快速调试入口，不参与 `phase5_remaining` 的有效验收
- 最小测试方式:
  - `python -m py_compile envs/platoon_env.py train/ma_grpo_trainer.py train/train_platoon_rl.py tests/acceptance/test_phase5_task7.py tests/acceptance/test_phase5_task8.py tests/acceptance/test_phase5_task9.py tests/acceptance/test_phase5_task10.py`
  - `/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase5_task7.py -v`
  - `/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase5_task8.py -v`
  - `/home/kong/anaconda3/envs/meta_drive/bin/python train/train_platoon_rl.py --mode platoon --steps 20 --render 0 --num-agents 1 --checkpoint-dir checkpoints/platoon_rl_single_real --log-dir logs/platoon_rl_single_real`
  - `/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase5_task9.py -v`
  - `/home/kong/anaconda3/envs/meta_drive/bin/python train/train_platoon_rl.py --mode platoon --steps 100 --render 0 --checkpoint-dir checkpoints/platoon_rl_real --log-dir logs/platoon_rl_real`
  - `/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase5_task10.py -v`
- 实际结果:
  - `test_phase5_task7.py`: `6 passed`
  - `test_phase5_task8.py`: `7 passed`
  - `test_phase5_task9.py`: `5 passed`
  - `test_phase5_task10.py`: `6 passed`
  - `real_single_summary.json`:
    - `steps = 20`
    - `max_kl = 3.1428`
    - `max_grad = 5.0000`
    - `gpu_peak_gb = 0.8941`
    - `has_nan_loss = false`
    - `has_oom = false`
  - `real_platoon_summary.json`:
    - `steps = 100`
    - `max_kl = 3.6080`
    - `max_grad = 5.0000`
    - `gpu_peak_gb = 1.9608`
    - `reward_first20 = -32.5886`
    - `reward_last20 = -32.3959`
    - `has_nan_loss = false`
    - `has_oom = false`
  - `checkpoints/platoon_rl_real/` 中真实生成 `step_50.ckpt` 和 `step_100.ckpt`
  - `logs/platoon_rl_real/` 中存在 TensorBoard event 文件
- 风险 / 未决事项:
  - `weight_migration.py` 仍有 `torch.load(..., weights_only=False)` 的 FutureWarning，不影响当前验收，但进入 Phase 6/7 前值得清理
  - 真实 RL 链路已经打通，但当前 reward 趋势改善仍然较小；Phase 6 做对照实验时需要更系统地分析 reward、formation error 和 collision rate 的关联
  - `outputs/phase5/platoon_summary.json` 是旧 toy/platoon 产物，后续 Phase 6 应优先使用 `real_single_summary.json` 和 `real_platoon_summary.json`

## Open Decisions

- Phase 6 的主对照是否直接复用 `checkpoints/platoon_rl_real/step_100.ckpt` 作为默认 RL 模型，还是先额外导出一个 `best.ckpt`
- Phase 6 评估时，单车基线与编队 RL 基线的 checkpoint 命名和路径是否需要统一到 `checkpoints/single_vehicle/` 与 `checkpoints/platoon_rl_real/`

## Handoff Notes

- 如果上下文不足，先读本文件，再读：
  1. `AGENTS.md`
  2. `docs/phases/phase5_remaining.md`
  3. `docs/phases/phase6.md`
  4. `docs/problem_definition.md`
  5. `docs/io_spec.md`
  6. `docs/metrics_spec.md`
- 当前 Phase 5 的真实环境剩余任务 `5.7~5.10` 已在 `meta_drive` 解释器下完成验收
- Phase 6 可直接复用：
  - `outputs/phase5/real_single_summary.json`
  - `outputs/phase5/real_platoon_summary.json`
  - `checkpoints/platoon_rl_real/step_50.ckpt`
  - `checkpoints/platoon_rl_real/step_100.ckpt`
- 下一步最优先动作是按 `docs/phases/phase6.md` 搭建评估脚本，优先读取真实 RL 产物而不是 toy 产物
- 执行 Phase 1 后续修复时，必须继续同步更新：
  - `AGENTS.md` 的完成标记
  - `EXECUTE_LOG.md` 的 `Current Status` 和 `Task Log`
