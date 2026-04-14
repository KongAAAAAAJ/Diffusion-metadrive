# EGO标记水平显示解决方案

## 问题描述
在 TopDown 视图中，当 `target_agent_heading_up=True` 时（摄像头随EGO车方向旋转），EGO标记（"EGO"文本）会跟随视角旋转，而不是始终保持水平显示。用户期望EGO标记和前车警告标记（"!"）一样，保持水平显示。

## 问题根本原因

### Canvas旋转管道
当 `target_agent_heading_up=True` 时，TopDownRenderer使用以下步骤：

1. **绘制所有对象到 `_frame_canvas`** - 地图、车辆等
2. **提取并旋转** - 将frame canvas的一部分复制到 `canvas_rotate`，然后整体旋转：
   ```python
   rotation = -np.rad2deg(v.heading_theta) + 90
   new_canvas = pygame.transform.rotozoom(self.canvas_rotate, rotation, 1)
   ```
3. **复制到屏幕** - 旋转后的canvas被blit到 `_screen_canvas`
4. **叠加标记** - EGO和warning标记被绘制到已旋转的screen_canvas上

### 标记绘制的问题
- EGO标记的**位置**通过 `_world_to_screen_position()` 被正确计算为旋转后的坐标
- 但标记的**文本本身**（"EGO"字符）没有被反向旋转
- 结果：文本看起来像是跟着背景一起旋转了

## 解决方案

### 核心思路
在绘制文本时应用**反向旋转**，使其始终保持水平。这是通过在文本被blit到canvas之前使用 `pygame.transform.rotate()` 实现的。

### 修改的文件

**文件**: `/metadrive/engine/top_down_renderer.py`

### 具体修改

#### 1. 修改 `_draw_tracked_agent_highlight()` 方法

添加 `heading_theta` 参数，并在绘制前对文本进行反向旋转：

```python
@classmethod
def _draw_tracked_agent_highlight(cls, surface, tracked_vehicle, screen_position=None, heading_theta=None) -> None:
    # ... 现有代码 ...
    text_surface = ObjectGraphics.font.render("EGO", True, cls.EGO_LABEL_FG, cls.EGO_LABEL_BG)
    
    # 在 target_agent_heading_up=True 时，应用反向旋转保持文本水平
    if heading_theta is not None:
        angle = -np.rad2deg(heading_theta) + 90.0
        text_surface = pygame.transform.rotate(text_surface, angle)
    
    # ... 绘制背景和文本 ...
```

#### 2. 修改 `_draw_warning_marker()` 方法

同样添加 `heading_theta` 参数和反向旋转逻辑：

```python
@classmethod
def _draw_warning_marker(cls, surface, tracked_vehicle, screen_position=None, heading_theta=None) -> None:
    # ... 现有代码 ...
    text_surface = ObjectGraphics.font.render("!", True, cls.WARNING_LABEL_FG, cls.WARNING_LABEL_BG)
    
    # 在 target_agent_heading_up=True 时，应用反向旋转保持文本水平
    if heading_theta is not None:
        angle = -np.rad2deg(heading_theta) + 90.0
        text_surface = pygame.transform.rotate(text_surface, angle)
    
    # ... 绘制背景和文本 ...
```

#### 3. 更新调用者

在 `_draw()` 方法中，当调用这两个方法时，传入 `heading_theta`：

```python
# EGO标记调用
ego_screen_position = self._world_to_screen_position(...)
if ego_screen_position is not None:
    heading_theta = self.current_track_agent.heading_theta if self.target_agent_heading_up else None
    self._draw_tracked_agent_highlight(
        self._screen_canvas,
        self.current_track_agent,
        screen_position=ego_screen_position,
        heading_theta=heading_theta,  # <- 传入heading_theta
    )

# Warning标记调用在 _draw_warning_markers() 中
def _draw_warning_markers(self, off) -> None:
    for vehicle in (self._latest_objects or {}).values():
        # ...
        heading_theta = self.current_track_agent.heading_theta if self.target_agent_heading_up else None
        self._draw_warning_marker(
            self._screen_canvas, 
            vehicle, 
            screen_position=screen_position, 
            heading_theta=heading_theta,  # <- 传入heading_theta
        )
```

## 技术细节

### 旋转角度计算
```python
angle = -np.rad2deg(heading_theta) + 90.0
```

这个公式与canvas旋转中使用的公式相同（见 `_draw()` 方法 L610），保证文本旋转与canvas旋转的协调一致。

### 向后兼容性
- 默认参数 `heading_theta=None` 确保不影响现有调用方
- 当 `heading_theta=None` 时，文本不被旋转（原有行为）
- 当 `heading_theta` 有值时，文本被反向旋转

### 行为总结
| `target_agent_heading_up` | 结果 |
|---------------------------|------|
| `False` | heading_theta=None → 文本不旋转（保持原有行为） |
| `True` | heading_theta=current_track_agent.heading_theta → 文本被反向旋转，始终水平 |

## 验证清单

- [x] 代码语法检查无误
- [x] 两个方法实现对称（EGO和warning行为一致）
- [x] 旋转角度计算与canvas旋转保持一致
- [x] 向后兼容（默认heading_theta=None）
- [ ] 运行时测试（推荐使用 `preview_scenario.sh` 并修改 `target_agent_heading_up=True` 进行测试）

## 测试建议

1. **修改测试配置**：在 `metadrive/exp_dataset/collect_expert.py` 中临时改为：
   ```python
   "target_agent_heading_up": True,  # 改为 True 用于测试
   ```

2. **运行预览脚本**：
   ```bash
   ./scripts/preview_scenario.sh S9_narrow_channel_negotiation 1
   ```

3. **验证效果**：
   - EGO标记应始终保持水平
   - Warning标记（!）也应始终保持水平
   - 整个背景和车辆应该随EGO车方向旋转

4. **恢复配置**：测试完成后改回：
   ```python
   "target_agent_heading_up": False,  # 改回 False
   ```

## 相关代码引用

- TopDownRenderer 类: `/metadrive/engine/top_down_renderer.py`
  - `_draw_tracked_agent_highlight()`: L452-481
  - `_draw_warning_marker()`: L483-512
  - `_draw_warning_markers()`: L675-685
  - `_draw()`: L514 onwards

- Canvas 旋转逻辑: `/metadrive/engine/top_down_renderer.py` L595-620
  - 关键: `rotation = -np.rad2deg(v.heading_theta) + 90`

## 潜在的后续优化

1. **性能考虑**：`pygame.transform.rotate()` 对每帧调用两次。如果性能成为问题，可以考虑缓存旋转后的文本表面（例如，对于"EGO"和"!"这样的固定字符串）。

2. **可配置旋转**：未来可能想要让标记旋转成为可配置选项，类似 `keep_markers_horizontal` 的参数。

3. **字体缩放**：旋转可能会因为像素化而影响渲染质量。可以考虑在旋转前使用更大的字体大小。
