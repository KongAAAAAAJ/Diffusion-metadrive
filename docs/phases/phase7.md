# Phase 7：论文材料生成

> **前置依赖**：Phase 6（评估结果）
> **后续 Phase**：无（终态）

## 目标
形成论文初稿所需全部材料。

---

## 子任务与验收指标

### ☐ 7.1 方法图与实验图表
- [ ] 待完成
- **交付物**：`paper/figures/` 下的图表文件
- **验收指标**（4 项）：
  1. 方法总览图（framework figure）数据/脚本
  2. 训练曲线图（reward、loss、formation_error 随 step）
  3. 场景对比可视化（≥ 3 场景 × 3 方法轨迹叠加）
  4. 分辨率 ≥ 300 DPI

### ☐ 7.2 论文大纲与数据整理
- [ ] 待完成
- **交付物**：`paper/outline.md`、`paper/tables/`、`paper/key_numbers.json`
- **验收指标**（4 项）：
  1. 大纲含 Abstract、Introduction、Related Work、Method、Experiments、Conclusion
  2. 每节 ≥ 3 要点
  3. 主表、消融表、泛化表 LaTeX 源码
  4. 关键数字提取到 `paper/key_numbers.json`

---

## 跨 Phase 数据流

```
Phase 5 logs/ (tensorboard) ──→ 训练曲线图
Phase 6 outputs/ ──→ 主表/消融表数据
Phase 6 tools/failure_analysis.py 输出 ──→ 失败案例图
```
