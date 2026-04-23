# Co-Preference 多车链路清理说明

本次重构后，多车编队链路唯一入口改为 co-preference target-point 生成，不再保留旧 selector mode 选择链路。

## 新链路

- 上层：`models/co_preference.CoPreferenceModel`
- 几何：`models/co_preference.geometry`
- 环境：`envs.co_preference_platoon_env.CoPreferencePlatoonEnv`
- 训练入口：`python -m train.train_co_preference`
- 配置：`configs/train/co_preference.yaml`
- 输出目录：`/media/kong/Elements_SE/Diffusion_Data/outputs/co_preference`

## 已移除旧链路

- `envs/selector_platoon_env.py`
- `models/selector/`
- `train/train_selector.py`
- `train/selector_callbacks.py`
- `configs/train/selector.yaml`
- 旧多车 checkpoint 可视化与 selector 训练脚本
- 旧 selector acceptance tests

## 单车链路保护

单车 diffusion 训练、开环评估和闭环测试入口保持原有 public behavior。单车内部的 multimodal candidates/logits/embedding 仍用于评估和可视化；多车 wrapper 不再将这些字段暴露为上层决策接口。
