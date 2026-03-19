1. 运行run_dataset_collect.sh 构建数据集
2. 运行abstract_anchors.py生成anchors
3. 默认可直接运行run_diffusion_train.sh 开始训练；脚本参数支持通过环境变量覆盖
4. `run_diffusion_preprocess.sh` 仅作为可选的离线特征缓存实验脚本，默认输出目录式 processed shard
5. `run_diffusion_convert_camera_layout.sh` 仅用于旧 CHW 相机数据迁移；新的 collect_expert 数据默认已是 HWC
6. legacy `.npz` processed dataset 不推荐继续生成；若坚持训练 processed dataset，优先使用目录式版本
