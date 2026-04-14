# EGO标记水平显示解决方案 - 快速参考

## 问题
TopDown视图中，当摄像头随EGO车旋转时(`target_agent_heading_up=True`)，"EGO"文本标记会旋转，而不是保持水平。

## 解决方案详细步骤

### 修改位置
**文件**: `/metadrive/engine/top_down_renderer.py`

### 三处关键修改

#### 修改1: `_draw_tracked_agent_highlight()` 方法 (L452-481)
**变更**: 添加 `heading_theta=None` 参数，并在绘制前反向旋转文本

```diff
- def _draw_tracked_agent_highlight(cls, surface, tracked_vehicle, screen_position=None) -> None:
+ def _draw_tracked_agent_highlight(cls, surface, tracked_vehicle, screen_position=None, heading_theta=None) -> None:
    # ... 现有代码 ...
    text_surface = ObjectGraphics.font.render("EGO", True, cls.EGO_LABEL_FG, cls.EGO_LABEL_BG)
+   
+   # 保持文本水平：当 target_agent_heading_up=True 时应用反向旋转
+   if heading_theta is not None:
+       angle = -np.rad2deg(heading_theta) + 90.0
+       text_surface = pygame.transform.rotate(text_surface, angle)
```

#### 修改2: `_draw_warning_marker()` 方法 (L483-512)
**变更**: 同样添加 `heading_theta=None` 参数和反向旋转逻辑

```diff
- def _draw_warning_marker(cls, surface, tracked_vehicle, screen_position=None) -> None:
+ def _draw_warning_marker(cls, surface, tracked_vehicle, screen_position=None, heading_theta=None) -> None:
    # ... 现有代码 ...
    text_surface = ObjectGraphics.font.render("!", True, cls.WARNING_LABEL_FG, cls.WARNING_LABEL_BG)
+   
+   # 保持文本水平：当 target_agent_heading_up=True 时应用反向旋转
+   if heading_theta is not None:
+       angle = -np.rad2deg(heading_theta) + 90.0
+       text_surface = pygame.transform.rotate(text_surface, angle)
```

#### 修改3: 调用方更新
在 `_draw()` 方法中传入 `heading_theta` 参数：

**EGO标记** (L638-649):
```diff
  if ego_screen_position is not None:
+     # 当 target_agent_heading_up=True 时传入 heading_theta 以保持水平
+     heading_theta = self.current_track_agent.heading_theta if self.target_agent_heading_up else None
      self._draw_tracked_agent_highlight(
          self._screen_canvas,
          self.current_track_agent,
          screen_position=ego_screen_position,
+         heading_theta=heading_theta,
      )
```

**Warning标记** (L676-683):
```diff
  def _draw_warning_markers(self, off) -> None:
      for vehicle in (self._latest_objects or {}).values():
          if getattr(vehicle, "scenario_warning_marker", None) != "!":
              continue
          screen_position = self._world_to_screen_position(vehicle.position, off)
          if screen_position is None:
              continue
+         # 当 target_agent_heading_up=True 时传入 heading_theta 以保持水平
+         heading_theta = self.current_track_agent.heading_theta if self.target_agent_heading_up else None
-         self._draw_warning_marker(self._screen_canvas, vehicle, screen_position=screen_position)
+         self._draw_warning_marker(self._screen_canvas, vehicle, screen_position=screen_position, heading_theta=heading_theta)
```

## 原理

### 旋转公式
```python
angle = -np.rad2deg(heading_theta) + 90.0
text_surface = pygame.transform.rotate(text_surface, angle)
```

这个公式与canvas旋转中使用的公式相同，使文本旋转相反方向，抵消背景的旋转。

### 结果
- 当 `target_agent_heading_up=False`: heading_theta=None，文本不旋转（原行为）
- 当 `target_agent_heading_up=True`: 文本被反向旋转，始终保持水平

## 验证

运行语法检查确认修改无误：
```bash
cd /home/kong/diffusion_codes/Diffusion-meta/Diffusion-metadrive
python -m py_compile metadrive/engine/top_down_renderer.py
# 无输出表示语法正确
```

## 测试

**快速测试**（临时修改配置）:

1. 编辑 `metadrive/exp_dataset/collect_expert.py` L492:
   ```python
   "target_agent_heading_up": True,  # 改为 True
   ```

2. 运行预览:
   ```bash
   ./scripts/preview_scenario.sh S9_narrow_channel_negotiation 1
   ```

3. 观察: EGO标记和"!"标记应保持水平

4. 恢复配置为 `False`

## 修改的代码差异摘要

| 部分 | 行号 | 修改内容 |
|------|------|----------|
| `_draw_tracked_agent_highlight()` 签名 | 454 | 添加 `heading_theta=None` |
| `_draw_tracked_agent_highlight()` 实现 | 474-477 | 添加反向旋转逻辑 |
| `_draw_warning_marker()` 签名 | 484 | 添加 `heading_theta=None` |
| `_draw_warning_marker()` 实现 | 506-509 | 添加反向旋转逻辑 |
| EGO标记调用 | 638-640 | 计算并传入heading_theta |
| Warning标记调用 | 676-678 | 计算并传入heading_theta |

## 关键要点

✅ **向后兼容**: 默认 `heading_theta=None`，不影响现有代码
✅ **对称实现**: EGO和警告标记行为一致
✅ **公式一致**: 使用与canvas旋转相同的角度计算
✅ **无性能问题**: `pygame.transform.rotate()` 每帧仅调用2次（EGO + warning）
