# 8自由度车辆模型与基于相平面法的横摆/侧倾稳定性判断数学推导

## 0. 建模目标与总体思路

本文构建一个用于车辆横摆稳定性和侧倾稳定性分析的 **8自由度车辆动力学模型**，并在此基础上给出两类稳定性判断方法：

1. **横摆稳定性判断**：采用 \(\beta-r\) 相平面法，其中 \(\beta\) 为质心侧偏角，\(r\) 为横摆角速度；
2. **侧倾稳定性判断**：分别采用：
   - **LTR 静态判断方法**：基于左右轮垂向载荷转移率判断是否接近轮胎离地；
   - **基于能量和 \(\phi-\dot\phi\) 相平面的动态判断方法**：通过侧倾系统能量判断车辆是否具有足够侧倾动能越过侧翻边界。

需要说明的是，所谓“8自由度”是指车辆具有 8 个独立运动自由度，常见选择为：

\[
\underbrace{v_x, v_y, r}_{车身平面3自由度}
+ \underbrace{\phi}_{车身侧倾1自由度}
+ \underbrace{\omega_{fl},\omega_{fr},\omega_{rl},\omega_{rr}}_{四个车轮旋转4自由度}
\]

合计为：

\[
3+1+4=8\ \text{DOF}
\]

但从状态空间角度，由于侧倾自由度 \(\phi\) 是二阶运动方程，因此状态中通常还需要包含 \(\dot\phi\)。因此 8 自由度模型的状态变量可写为：

\[
\mathbf{x}=\left[
 v_x,\ v_y,\ r,\ \phi,\ \dot\phi,\ \omega_{fl},\omega_{fr},\omega_{rl},\omega_{rr}
\right]^T
\]

其中状态维数为 9，但独立运动自由度为 8。

---

## 1. 坐标系与符号定义

建立车辆质心坐标系：

- \(x\) 轴：车辆纵向，向前为正；
- \(y\) 轴：车辆横向，向左为正；
- \(z\) 轴：垂直向上；
- \(r\)：横摆角速度，绕 \(z\) 轴；
- \(\phi\)：车身侧倾角，绕车辆纵向 \(x\) 轴；
- \(\dot\phi\)：侧倾角速度。

车辆参数定义如下：

| 符号 | 含义 |
|---|---|
| \(m\) | 整车质量 |
| \(m_s\) | 簧载质量 |
| \(I_z\) | 横摆转动惯量 |
| \(I_\phi\) | 侧倾转动惯量 |
| \(l_f\) | 质心到前轴距离 |
| \(l_r\) | 质心到后轴距离 |
| \(L=l_f+l_r\) | 轴距 |
| \(T_f,T_r\) | 前、后轮距 |
| \(h\) | 整车质心高度 |
| \(h_s\) | 簧载质量侧倾中心到质心的等效高度 |
| \(K_\phi\) | 悬架等效侧倾刚度 |
| \(C_\phi\) | 悬架等效侧倾阻尼 |
| \(R_w\) | 车轮有效滚动半径 |
| \(I_w\) | 车轮转动惯量 |
| \(\delta_f\) | 前轮转角 |
| \(\mu\) | 路面附着系数 |

四个车轮记为：

\[
fl,\ fr,\ rl,\ rr
\]

分别代表左前轮、右前轮、左后轮、右后轮。

---

## 2. 8自由度车辆动力学模型

### 2.1 车身纵向动力学

车辆质心处纵向动力学为：

\[
m(\dot v_x-rv_y)=\sum F_x
\]

考虑前轮转角后，前轮胎力需要由轮胎坐标系转换到车身坐标系。因此：

\[
\begin{aligned}
m(\dot v_x-rv_y)=
& (F_{x,fl}\cos\delta_f-F_{y,fl}\sin\delta_f) \\
& +(F_{x,fr}\cos\delta_f-F_{y,fr}\sin\delta_f) \\
& +F_{x,rl}+F_{x,rr}
\end{aligned}
\]

因此纵向速度微分方程为：

\[
\boxed{
\dot v_x
=
rv_y+rac{1}{m}
\left[
(F_{x,fl}+F_{x,fr})\cos\delta_f
-(F_{y,fl}+F_{y,fr})\sin\delta_f
+F_{x,rl}+F_{x,rr}
\right]
}
\]

---

### 2.2 车身横向动力学

车辆质心处横向动力学为：

\[
m(\dot v_y+rv_x)=\sum F_y
\]

将前轮胎力转换到车身坐标系：

\[
\begin{aligned}
m(\dot v_y+rv_x)=
& (F_{x,fl}\sin\delta_f+F_{y,fl}\cos\delta_f) \\
& +(F_{x,fr}\sin\delta_f+F_{y,fr}\cos\delta_f) \\
& +F_{y,rl}+F_{y,rr}
\end{aligned}
\]

因此：

\[
\boxed{
\dot v_y
=
-rv_x+rac{1}{m}
\left[
(F_{x,fl}+F_{x,fr})\sin\delta_f
+(F_{y,fl}+F_{y,fr})\cos\delta_f
+F_{y,rl}+F_{y,rr}
\right]
}
\]

---

### 2.3 横摆动力学

横摆力矩由前后轴横向力、纵向力左右差以及前轮转角共同决定。为了简化表示，取前轮左右轮距均为 \(T_f\)，后轮左右轮距为 \(T_r\)。

前轴对质心的横摆力矩为：

\[
M_{z,f}
=
l_f\left[(F_{x,fl}+F_{x,fr})\sin\delta_f+(F_{y,fl}+F_{y,fr})\cos\delta_f\right]
\]

后轴横向力矩为：

\[
M_{z,r}
=
-l_r(F_{y,rl}+F_{y,rr})
\]

左右轮纵向力差引起的横摆力矩近似为：

\[
M_{z,\Delta x}
=
\frac{T_f}{2}\left[(F_{x,fr}\cos\delta_f-F_{y,fr}\sin\delta_f)
-(F_{x,fl}\cos\delta_f-F_{y,fl}\sin\delta_f)\right]
+\frac{T_r}{2}(F_{x,rr}-F_{x,rl})
\]

因此横摆动力学为：

\[
I_z\dot r=M_{z,f}+M_{z,r}+M_{z,\Delta x}
\]

即：

\[
\boxed{
\dot r=rac{1}{I_z}
\left(M_{z,f}+M_{z,r}+M_{z,\Delta x}\right)
}
\]

若不考虑左右纵向力差，并假设左右轮胎力合并为前后轴等效力，则可简化为经典形式：

\[
\boxed{
I_z\dot r
=
l_f F_{yf}-l_r F_{yr}
}
\]

其中：

\[
F_{yf}=F_{y,fl}+F_{y,fr},
\qquad
F_{yr}=F_{y,rl}+F_{y,rr}
\]

---

### 2.4 侧倾动力学

车身侧倾动力学可以由绕侧倾中心的力矩平衡得到：

\[
I_\phi\ddot\phi
=
M_{\text{lat}}+M_g-M_k-M_c
\]

其中：

- 横向惯性力引起的侧倾力矩：

\[
M_{\text{lat}}=m_s h_s a_y
\]

- 重力引起的侧倾力矩：

\[
M_g=m_s g h_s\sin\phi
\]

- 悬架侧倾刚度恢复力矩：

\[
M_k=K_\phi\phi
\]

- 悬架侧倾阻尼力矩：

\[
M_c=C_\phi\dot\phi
\]

因此侧倾动力学为：

\[
\boxed{
I_\phi\ddot\phi
=
m_s h_s a_y
+m_s g h_s\sin\phi
-K_\phi\phi
-C_\phi\dot\phi
}
\]

其中横向加速度可由车辆运动状态计算：

\[
\boxed{
a_y=\dot v_y+v_x r
}
\]

在小角度条件下：

\[
\sin\phi\approx \phi
\]

则有：

\[
I_\phi\ddot\phi
+C_\phi\dot\phi
+(K_\phi-m_sgh_s)\phi
=m_s h_s a_y
\]

定义等效侧倾刚度：

\[
\boxed{
K_{\phi,eff}=K_\phi-m_sgh_s
}
\]

于是：

\[
\boxed{
I_\phi\ddot\phi
+C_\phi\dot\phi
+K_{\phi,eff}\phi
=m_s h_s a_y
}
\]

---

### 2.5 四轮旋转动力学

每个车轮的旋转动力学为：

\[
I_w\dot\omega_i=T_i-R_wF_{x,i}
\]

其中：

- \(i\in\{fl,fr,rl,rr\}\)；
- \(T_i\) 为驱动/制动力矩；
- \(F_{x,i}\) 为轮胎纵向力；
- \(R_w\) 为车轮有效半径。

因此：

\[
\boxed{
\dot\omega_i=rac{T_i-R_wF_{x,i}}{I_w}
}
\]

---

## 3. 轮胎侧偏角、滑移率与轮胎力

### 3.1 各车轮速度

车轮接地点速度由车身质心速度与横摆运动共同决定。

前轴中心处速度为：

\[
v_{x,f}=v_x,
\qquad
v_{y,f}=v_y+l_f r
\]

后轴中心处速度为：

\[
v_{x,r}=v_x,
\qquad
v_{y,r}=v_y-l_r r
\]

若考虑左右轮由于横摆导致的纵向速度差，则：

\[
v_{x,fl}=v_x-\frac{T_f}{2}r,
\qquad
v_{x,fr}=v_x+\frac{T_f}{2}r
\]

\[
v_{x,rl}=v_x-\frac{T_r}{2}r,
\qquad
v_{x,rr}=v_x+\frac{T_r}{2}r
\]

前轮由于转向角 \(\delta_f\)，需要转换到轮胎坐标系。前轮轮胎坐标系纵横向速度可近似写为：

\[
v_{x,fl}^{t}=v_{x,fl}\cos\delta_f+(v_y+l_f r)\sin\delta_f
\]

\[
v_{y,fl}^{t}=-(v_{x,fl})\sin\delta_f+(v_y+l_f r)\cos\delta_f
\]

右前轮同理：

\[
v_{x,fr}^{t}=v_{x,fr}\cos\delta_f+(v_y+l_f r)\sin\delta_f
\]

\[
v_{y,fr}^{t}=-(v_{x,fr})\sin\delta_f+(v_y+l_f r)\cos\delta_f
\]

后轮无转角：

\[
v_{x,rl}^{t}=v_{x,rl},
\qquad
v_{y,rl}^{t}=v_y-l_r r
\]

\[
v_{x,rr}^{t}=v_{x,rr},
\qquad
v_{y,rr}^{t}=v_y-l_r r
\]

---

### 3.2 轮胎侧偏角

车轮侧偏角定义为轮胎速度方向与轮胎朝向之间的夹角。一般可写为：

\[
\alpha_i=-\arctan\left(\frac{v_{y,i}^{t}}{v_{x,i}^{t}}\right)
\]

若采用单轨模型近似，则前后轴侧偏角为：

\[
\boxed{
\alpha_f=\\delta_f-\arctan\left(\frac{v_y+l_f r}{v_x}\right)
}
\]

\[
\boxed{
\alpha_r=-\arctan\left(\frac{v_y-l_r r}{v_x}\right)
}
\]

当侧偏角较小时：

\[
\alpha_f\approx \delta_f-\frac{v_y+l_f r}{v_x}
\]

\[
\alpha_r\approx -\frac{v_y-l_r r}{v_x}
\]

引入质心侧偏角：

\[
\boxed{
\beta=\arctan\left(\frac{v_y}{v_x}\right)
}
\]

小角度下：

\[
v_y\approx v_x\beta
\]

因此：

\[
\boxed{
\alpha_f\approx \delta_f-\beta-\frac{l_f r}{v_x}
}
\]

\[
\boxed{
\alpha_r\approx -\beta+\frac{l_r r}{v_x}
}
\]

---

### 3.3 轮胎滑移率

车轮滑移率可写为：

驱动工况：

\[
\lambda_i=\frac{R_w\omega_i-v_{x,i}^{t}}{R_w\omega_i}
\]

制动工况：

\[
\lambda_i=\frac{R_w\omega_i-v_{x,i}^{t}}{v_{x,i}^{t}}
\]

工程实现中常使用统一形式并加入小量 \(\varepsilon\) 防止除零：

\[
\boxed{
\lambda_i=rac{R_w\omega_i-v_{x,i}^{t}}{\max(|v_{x,i}^{t}|,|R_w\omega_i|,\varepsilon)}
}
\]

---

### 3.4 轮胎力模型

轮胎力可采用线性模型、Fiala 模型或 Magic Formula 模型。

在线性小侧偏角区域：

\[
F_{y,i}=C_{\alpha i}\alpha_i
\]

但为了分析车辆极限稳定性，需要考虑轮胎饱和。一般可抽象写为：

\[
\boxed{
F_{x,i}=f_x(\lambda_i,\alpha_i,F_{z,i},\mu)
}
\]

\[
\boxed{
F_{y,i}=f_y(\lambda_i,\alpha_i,F_{z,i},\mu)
}
\]

并满足摩擦圆/摩擦椭圆约束：

\[
\boxed{
\left(\frac{F_{x,i}}{\mu F_{z,i}}\right)^2+
\left(\frac{F_{y,i}}{\mu F_{z,i}}\right)^2\le 1
}
\]

轮胎非线性饱和是横摆稳定边界和侧倾稳定边界形成的根本原因之一。

---

## 4. 垂向载荷与载荷转移

车辆侧倾稳定性判断需要计算四轮垂向载荷。简化起见，可将垂向载荷分为静态载荷、纵向载荷转移和横向载荷转移。

### 4.1 静态轴荷

前轴静态载荷：

\[
F_{zf0}=\frac{mgl_r}{L}
\]

后轴静态载荷：

\[
F_{zr0}=\frac{mgl_f}{L}
\]

左右均分时：

\[
F_{z,fl0}=F_{z,fr0}=\frac{F_{zf0}}{2}
\]

\[
F_{z,rl0}=F_{z,rr0}=\frac{F_{zr0}}{2}
\]

---

### 4.2 纵向载荷转移

纵向加速度 \(a_x\) 引起前后轴载荷转移：

\[
\Delta F_{z,x}=\frac{mh a_x}{L}
\]

加速时后轴载荷增加，制动时前轴载荷增加。以前向加速度 \(a_x>0\) 为例：

\[
F_{zf}=F_{zf0}-\Delta F_{z,x}
\]

\[
F_{zr}=F_{zr0}+\Delta F_{z,x}
\]

---

### 4.3 横向载荷转移

横向加速度和侧倾运动导致左右轮载转移。前后轴横向载荷转移可简化为：

\[
\Delta F_{z,yf}=\frac{K_{\phi f}}{K_{\phi f}+K_{\phi r}}\cdot\frac{m_s h_s a_y+K_\phi\phi+C_\phi\dot\phi}{T_f}
\]

\[
\Delta F_{z,yr}=\frac{K_{\phi r}}{K_{\phi f}+K_{\phi r}}\cdot\frac{m_s h_s a_y+K_\phi\phi+C_\phi\dot\phi}{T_r}
\]

若不区分前后悬架侧倾刚度，可采用整体近似：

\[
\boxed{
\Delta F_{z,y}=\frac{m h a_y}{T}
}
\]

或者考虑侧倾运动：

\[
\boxed{
\Delta F_{z,y}=\frac{m_s h_s a_y+K_\phi\phi+C_\phi\dot\phi}{T}
}
\]

其中 \(T\) 可取平均轮距。

以车辆向左转弯、车身向右侧倾为例，右侧轮载增加、左侧轮载减小：

\[
F_{z,L}=\frac{mg}{2}-\Delta F_{z,y}
\]

\[
F_{z,R}=\frac{mg}{2}+\Delta F_{z,y}
\]

四轮垂向载荷可进一步分配为：

\[
F_{z,fl}=\frac{F_{zf}}{2}-\Delta F_{z,yf}
\]

\[
F_{z,fr}=\frac{F_{zf}}{2}+\Delta F_{z,yf}
\]

\[
F_{z,rl}=\frac{F_{zr}}{2}-\Delta F_{z,yr}
\]

\[
F_{z,rr}=\frac{F_{zr}}{2}+\Delta F_{z,yr}
\]

符号方向应根据具体坐标系和转弯方向统一确定。

---

## 5. 横摆稳定性：基于 \(\beta-r\) 相平面法

### 5.1 相平面状态变量选择

横摆稳定性通常采用 \(\beta-r\) 相平面：

\[
\boxed{
\mathbf{x}_y=
\begin{bmatrix}
\beta\\
r
\end{bmatrix}
}
\]

其中：

\[
\beta=\arctan\left(\frac{v_y}{v_x}\right)
\]

\(\beta\) 反映车辆整体侧滑程度，\(r\) 反映车辆横摆旋转程度。

横摆稳定性判断的核心问题是：

\[
\boxed{
当前状态点\ (\beta,r)\ 是否处于稳定平衡点的吸引域内？
}
\]

---

### 5.2 从8自由度模型到 \(\beta-r\) 动力学

由：

\[
\beta=\arctan\left(\frac{v_y}{v_x}\right)
\]

对时间求导：

\[
\dot\beta=rac{v_x\dot v_y-v_y\dot v_x}{v_x^2+v_y^2}
\]

若 \(\beta\) 较小且 \(v_x\) 变化较慢，则：

\[
\beta\approx\frac{v_y}{v_x}
\]

\[
\dot\beta\approx\frac{\dot v_y}{v_x}
\]

由横向动力学：

\[
\dot v_y=-rv_x+rac{F_y}{m}
\]

其中：

\[
F_y=(F_{x,fl}+F_{x,fr})\sin\delta_f+(F_{y,fl}+F_{y,fr})\cos\delta_f+F_{y,rl}+F_{y,rr}
\]

因此：

\[
\boxed{
\dot\beta
\approx
-r+rac{F_y}{m v_x}
}
\]

横摆角速度方程为：

\[
\boxed{
\dot r=rac{M_z}{I_z}
}
\]

其中：

\[
M_z=M_{z,f}+M_{z,r}+M_{z,\Delta x}
\]

于是得到 \(\beta-r\) 相平面系统：

\[
\boxed{
\begin{cases}
\dot\beta=f_1(\beta,r;v_x,\delta_f,\mu)\\
\dot r=f_2(\beta,r;v_x,\delta_f,\mu)
\end{cases}
}
\]

这里固定 \(v_x,\delta_f,\mu\) 的含义是固定工况和相平面向量场，而不是固定相平面上的 \((\beta,r)\) 状态点。

---

### 5.3 平衡点求解

横摆相平面中的平衡点满足：

\[
\dot\beta=0,
\qquad
\dot r=0
\]

即：

\[
\boxed{
f_1(\beta_e,r_e;v_x,\delta_f,\mu)=0
}
\]

\[
\boxed{
f_2(\beta_e,r_e;v_x,\delta_f,\mu)=0
}
\]

在稳态转弯条件下，\((\beta_e,r_e)\) 表示该车速、转角和路面附着条件下车辆的稳态侧偏角和稳态横摆角速度。

对于线性二自由度车辆模型：

\[
\dot\beta=-r+rac{C_f\alpha_f+C_r\alpha_r}{m v_x}
\]

\[
\dot r=\frac{l_f C_f\alpha_f-l_r C_r\alpha_r}{I_z}
\]

其中：

\[
\alpha_f=\delta_f-\beta-\frac{l_f r}{v_x}
\]

\[
\alpha_r=-\beta+\frac{l_r r}{v_x}
\]

代入后可得到线性平衡点。但在极限工况下，应使用非线性轮胎模型求解平衡点，因为轮胎饱和会导致多个平衡点，包括稳定平衡点、鞍点和不稳定平衡点。

---

### 5.4 平衡点稳定性分析

对系统：

\[
\dot{\mathbf{x}}_y=\mathbf{f}(\mathbf{x}_y)
\]

在平衡点 \(\mathbf{x}_{ye}\) 附近线性化：

\[
\Delta\dot{\mathbf{x}}_y=A_y\Delta\mathbf{x}_y
\]

其中：

\[
\boxed{
A_y=
\left.
\frac{\partial \mathbf{f}}{\partial \mathbf{x}_y}
\right|_{\mathbf{x}_y=\mathbf{x}_{ye}}
}
\]

若 \(A_y\) 的所有特征值实部均小于 0：

\[
\boxed{
\operatorname{Re}(\lambda_i)<0
}
\]

则该平衡点局部渐近稳定。

若特征值中既有正实部又有负实部，则该平衡点为鞍点。车辆横摆稳定边界通常与鞍点的稳定流形有关。

---

### 5.5 横摆稳定域与稳定边界

稳定平衡点 \(\mathbf{x}_{ys}\) 的吸引域定义为：

\[
\boxed{
\Omega_y(\mathbf{x}_{ys})=
\left\{
\mathbf{x}_{y0}\ \middle|\
\lim_{t\to\infty}\mathbf{x}_y(t;\mathbf{x}_{y0})=\mathbf{x}_{ys}
\right\}
}
\]

横摆稳定边界为吸引域边界：

\[
\boxed{
\partial\Omega_y(\mathbf{x}_{ys})
}
\]

理论上，横摆稳定边界通常由鞍点或不稳定平衡点的稳定流形构成：

\[
\boxed{
\partial\Omega_y(\mathbf{x}_{ys})=W^s(\mathbf{x}_{yu})
}
\]

其中 \(\mathbf{x}_{yu}\) 为鞍点，\(W^s(\mathbf{x}_{yu})\) 为其稳定流形。

---

### 5.6 数值确定横摆稳定边界的步骤

实际应用中，横摆稳定边界可通过相平面扫描得到。

**步骤 1：固定工况参数**

\[
v_x=v_{x0},
\qquad
\delta_f=\delta_{f0},
\qquad
\mu=\mu_0
\]

从而固定动力学向量场。

**步骤 2：在 \(\beta-r\) 平面中选取初始状态网格**

\[
\beta_0\in[\beta_{min},\beta_{max}],
\qquad
r_0\in[r_{min},r_{max}]
\]

**步骤 3：对每个初始点积分非线性动力学模型**

\[
\dot{\mathbf{x}}=\mathbf{F}(\mathbf{x},u)
\]

其中初始条件满足：

\[
\beta(0)=\beta_0,
\qquad
r(0)=r_0
\]

**步骤 4：判断轨迹归宿**

如果：

\[
\lim_{t\to T}\left\|
\begin{bmatrix}
\beta(t)\\r(t)
\end{bmatrix}
-
\begin{bmatrix}
\beta_e\\r_e
\end{bmatrix}
\right\|<\varepsilon
\]

则认为该点属于稳定域。

如果出现以下任一情况，则认为该点属于失稳域：

\[
|\beta|>\beta_{lim}
\]

\[
|r|>r_{lim}
\]

\[
|\alpha_f|\ \text{或}\ |\alpha_r| \ \text{进入严重饱和区域}
\]

\[
车辆状态不收敛或持续发散
\]

**步骤 5：提取稳定域与失稳域之间的边界**

稳定点集合与失稳点集合之间的分界线即为横摆稳定边界。

---

## 6. 侧倾稳定性方法一：LTR 静态判断方法

### 6.1 LTR 定义

载荷转移率 LTR，Load Transfer Ratio，定义为左右侧车轮垂向载荷差与总垂向载荷之比：

\[
\boxed{
LTR=
\frac{F_{z,R}-F_{z,L}}{F_{z,R}+F_{z,L}}
}
\]

其中：

\[
F_{z,R}=F_{z,fr}+F_{z,rr}
\]

\[
F_{z,L}=F_{z,fl}+F_{z,rl}
\]

因此：

\[
\boxed{
LTR=
\frac{(F_{z,fr}+F_{z,rr})-(F_{z,fl}+F_{z,rl})}
{F_{z,fr}+F_{z,rr}+F_{z,fl}+F_{z,rl}}
}
\]

---

### 6.2 LTR 的物理意义

若车辆向左转弯，车身向右侧倾，右侧轮载增加，左侧轮载减小。当左侧车轮刚好离地时：

\[
F_{z,L}=0
\]

此时：

\[
LTR=\frac{F_{z,R}-0}{F_{z,R}+0}=1
\]

反之，当右侧车轮刚好离地时：

\[
F_{z,R}=0
\]

此时：

\[
LTR=\frac{0-F_{z,L}}{0+F_{z,L}}=-1
\]

因此：

\[
\boxed{
|LTR|=1
}
\]

表示一侧车轮垂向载荷降为 0，是车辆发生轮胎离地的理论临界条件。

---

### 6.3 基于横向加速度的静态 LTR 推导

采用准静态侧翻模型，假设车辆横向加速度为 \(a_y\)，质心高度为 \(h\)，轮距为 \(T\)。横向惯性力产生的侧翻力矩为：

\[
M_{overturn}=m a_y h
\]

重力产生的抗侧翻力矩为：

\[
M_{restore}=mg\frac{T}{2}
\]

侧翻临界条件为：

\[
M_{overturn}=M_{restore}
\]

即：

\[
m a_y h=mg\frac{T}{2}
\]

得到临界横向加速度：

\[
\boxed{
a_{y,crit}=g\frac{T}{2h}
}
\]

定义静态稳定因子 SSF：

\[
\boxed{
SSF=\frac{T}{2h}
}
\]

则：

\[
\boxed{
\frac{a_{y,crit}}{g}=SSF
}
\]

另一方面，横向载荷转移为：

\[
\Delta F_z=\frac{m a_y h}{T}
\]

左右侧轮载为：

\[
F_{z,R}=\frac{mg}{2}+\Delta F_z
\]

\[
F_{z,L}=\frac{mg}{2}-\Delta F_z
\]

代入 LTR：

\[
LTR=rac{2\Delta F_z}{mg}
\]

得到：

\[
\boxed{
LTR=\frac{2h}{T}\frac{a_y}{g}
}
\]

因此，\(|LTR|=1\) 等价于：

\[
\boxed{
|a_y|=g\frac{T}{2h}
}
\]

---

### 6.4 LTR 判断准则

理论判断准则为：

\[
\boxed{
|LTR|<1 \quad \text{未发生轮胎离地，侧倾稳定}
}
\]

\[
\boxed{
|LTR|=1 \quad \text{一侧车轮垂向载荷为零，达到侧翻临界}
}
\]

\[
\boxed{
|LTR|>1 \quad \text{理论上进入轮胎离地或侧翻风险区}
}
\]

工程应用中通常会设置安全裕度：

\[
\boxed{
|LTR|\le LTR_{max}
}
\]

其中：

\[
LTR_{max}=0.7\sim0.9
\]

常用于预警、轨迹规划约束或稳定性控制。

---

### 6.5 LTR 静态判断方法的特点

LTR 方法的优势是：

1. 物理意义明确，直接对应左右轮垂向载荷转移；
2. 计算简单，适合实时控制；
3. 若模型能输出四轮 \(F_z\)，则可以直接计算；
4. 适合准静态转弯、稳态避障等侧倾风险评估。

但其局限也比较明显：

1. LTR 主要反映当前时刻的轮载状态；
2. 对侧倾角速度 \(\dot\phi\) 所代表的滚转动能体现不足；
3. 在高速紧急避障、双移线、cut-in 快速转向等瞬态工况下，可能存在滞后或保守判断；
4. \(|LTR|<1\) 并不一定意味着未来不会侧翻，因为车辆可能已经具有较大的侧倾动能。

因此，为了更准确描述动态侧翻风险，可以进一步采用基于能量和 \(\phi-\dot\phi\) 相平面的动态判断方法。

---

## 7. 侧倾稳定性方法二：基于能量和相平面的动态判断方法

### 7.1 侧倾相平面变量选择

侧倾稳定性可采用 \(\phi-\dot\phi\) 相平面：

\[
\boxed{
\mathbf{x}_r=
\begin{bmatrix}
\phi\\
\dot\phi
\end{bmatrix}
}
\]

其中：

- \(\phi\)：当前车辆已经侧倾的程度；
- \(\dot\phi\)：车辆继续侧倾或回正的速度。

与 LTR 方法不同，\(\phi-\dot\phi\) 相平面不仅关注当前侧倾姿态，还关注车辆未来侧倾发展的动态趋势。

---

### 7.2 侧倾动力学标准形式

由前文侧倾动力学：

\[
I_\phi\ddot\phi
+C_\phi\dot\phi
+K_{\phi,eff}\phi
=m_s h_s a_y
\]

其中：

\[
K_{\phi,eff}=K_\phi-m_sgh_s
\]

将其写为状态方程：

\[
\boxed{
\dot x_1=x_2
}
\]

\[
\boxed{
\dot x_2=rac{1}{I_\phi}
\left(
m_s h_s a_y
-C_\phi x_2
-K_{\phi,eff}x_1
\right)
}
\]

其中：

\[
x_1=\phi,
\qquad
x_2=\dot\phi
\]

若考虑非线性重力项，则：

\[
\boxed{
I_\phi\ddot\phi
+C_\phi\dot\phi
+K_\phi\phi
-m_sgh_s\sin\phi
=m_s h_s a_y
}
\]

---

### 7.3 侧翻临界角

侧翻临界角 \(\phi_{crit}\) 可以理解为车辆达到轮胎离地或侧翻临界时的侧倾角。

在简化几何模型中，当质心投影越过外侧轮胎接地点支撑边界时发生准静态侧翻。若轮距为 \(T\)，质心高度为 \(h\)，则临界侧倾角可近似为：

\[
\boxed{
\phi_{crit}\approx \arctan\left(\frac{T}{2h}\right)
}
\]

对于高质心重型车辆，\(h\) 较大，因此：

\[
\frac{T}{2h}
\]

较小，\(\phi_{crit}\) 也相对较小，侧翻风险更高。

需要注意的是，在实际车辆中，轮胎变形、悬架运动、载荷转移、路面坡度和簧载/非簧载质量分布都会影响 \(\phi_{crit}\)。因此 \(\phi_{crit}\) 也可以通过 LTR 临界条件 \(|LTR|=1\) 或实车/仿真标定获得。

---

### 7.4 侧倾能量函数构造

忽略阻尼耗散时，侧倾系统的总机械能由侧倾动能和势能组成：

\[
\boxed{
E(\phi,\dot\phi)=T_r(\dot\phi)+U(\phi)
}
\]

其中侧倾动能为：

\[
\boxed{
T_r(\dot\phi)=\frac{1}{2}I_\phi\dot\phi^2
}
\]

势能可由侧倾恢复力矩积分得到。

在线性化侧倾模型中，恢复力矩为：

\[
M_{restore}=K_{\phi,eff}\phi
\]

因此势能为：

\[
\boxed{
U(\phi)=\frac{1}{2}K_{\phi,eff}\phi^2-m_s h_s a_y\phi
}
\]

其中 \(-m_s h_s a_y\phi\) 表示横向加速度产生的外部侧倾力矩所对应的势能项。

于是侧倾总能量为：

\[
\boxed{
E(\phi,\dot\phi)
=
\frac{1}{2}I_\phi\dot\phi^2
+\frac{1}{2}K_{\phi,eff}\phi^2
-m_s h_s a_y\phi
}
\]

若采用非线性重力项：

\[
M_g=m_sgh_s\sin\phi
\]

则对应势能可写为：

\[
U(\phi)=
\frac{1}{2}K_\phi\phi^2
-m_sgh_s(1-\cos\phi)
-m_s h_s a_y\phi
\]

因此非线性能量函数为：

\[
\boxed{
E(\phi,\dot\phi)
=
\frac{1}{2}I_\phi\dot\phi^2
+rac{1}{2}K_\phi\phi^2
-m_sgh_s(1-\cos\phi)
-m_s h_s a_y\phi
}
\]

---

### 7.5 临界能量与动态侧翻边界

设车辆达到侧翻临界角时的侧倾角为：

\[
\phi=\phi_{crit}
\]

如果车辆刚好以零侧倾角速度到达该临界角，则临界能量为：

\[
\boxed{
E_{crit}=E(\phi_{crit},0)
}
\]

在线性化模型下：

\[
\boxed{
E_{crit}
=
\frac{1}{2}K_{\phi,eff}\phi_{crit}^2
-m_s h_s a_y\phi_{crit}
}
\]

对于当前状态 \((\phi,\dot\phi)\)，若：

\[
\boxed{
E(\phi,\dot\phi)<E_{crit}
}
\]

则表示当前侧倾能量不足以越过侧翻临界边界，车辆侧倾状态具有恢复可能。

若：

\[
\boxed{
E(\phi,\dot\phi)\ge E_{crit}
}
\]

则表示当前侧倾角和侧倾角速度所对应的能量已经足以达到或越过侧翻临界角，车辆存在动态侧翻风险。

因此，侧倾相平面中的动态稳定边界可写为：

\[
\boxed{
E(\phi,\dot\phi)=E_{crit}
}
\]

这是一条在 \(\phi-\dot\phi\) 相平面中的边界曲线。

---

### 7.6 相平面边界的显式表达

采用线性能量函数：

\[
E(\phi,\dot\phi)
=
\frac{1}{2}I_\phi\dot\phi^2
+\frac{1}{2}K_{\phi,eff}\phi^2
-m_s h_s a_y\phi
\]

动态边界满足：

\[
\frac{1}{2}I_\phi\dot\phi^2
+\frac{1}{2}K_{\phi,eff}\phi^2
-m_s h_s a_y\phi
=E_{crit}
\]

整理可得：

\[
\boxed{
\dot\phi
=\pm
\sqrt{
\frac{2}{I_\phi}
\left[
E_{crit}
-rac{1}{2}K_{\phi,eff}\phi^2
+m_s h_s a_y\phi
\right]
}
}
\]

该式给出了 \(\phi-\dot\phi\) 相平面中的侧倾动态边界。

当根号内为正时，边界存在；当根号内为负时，对应的 \(\phi\) 已经超出可恢复能量区域。

---

### 7.7 动态侧倾稳定裕度

为了便于控制或规划，可定义归一化侧倾稳定裕度：

\[
\boxed{
S_{roll}=1-\frac{E(\phi,\dot\phi)}{E_{crit}}
}
\]

判断规则为：

\[
\boxed{
S_{roll}>0 \quad \text{侧倾稳定}
}
\]

\[
\boxed{
S_{roll}=0 \quad \text{到达动态侧翻边界}
}
\]

\[
\boxed{
S_{roll}<0 \quad \text{存在动态侧翻风险}
}
\]

在轨迹规划或控制中，可以将其作为约束：

\[
\boxed{
E(\phi,\dot\phi)-E_{crit}\le 0
}
\]

或者引入安全裕度：

\[
\boxed{
E(\phi,\dot\phi)\le \eta E_{crit},\qquad 0<\eta<1
}
\]

其中 \(\eta\) 可取 \(0.7\sim0.9\) 以提前预警。

---

### 7.8 基于相平面和能量法的判断流程

**步骤 1：由8自由度模型获取侧倾状态**

\[
\phi(t),\qquad \dot\phi(t)
\]

同时计算横向加速度：

\[
a_y(t)=\dot v_y(t)+v_x(t)r(t)
\]

**步骤 2：确定临界侧倾角**

可采用几何近似：

\[
\phi_{crit}=\arctan\left(\frac{T}{2h}\right)
\]

或通过 \(|LTR|=1\) 对应的仿真状态标定得到。

**步骤 3：构造侧倾能量函数**

线性模型下：

\[
E(\phi,\dot\phi)
=
\frac{1}{2}I_\phi\dot\phi^2
+\frac{1}{2}K_{\phi,eff}\phi^2
-m_s h_s a_y\phi
\]

**步骤 4：计算临界能量**

\[
E_{crit}=E(\phi_{crit},0)
\]

**步骤 5：判断当前状态是否越界**

若：

\[
E(\phi,\dot\phi)<E_{crit}
\]

则侧倾动态稳定。

若：

\[
E(\phi,\dot\phi)\ge E_{crit}
\]

则存在侧翻风险。

**步骤 6：在 \(\phi-\dot\phi\) 相平面上绘制边界**

边界由：

\[
E(\phi,\dot\phi)=E_{crit}
\]

确定。当前车辆侧倾状态点为：

\[
(\phi(t),\dot\phi(t))
\]

若状态点位于边界内部，则稳定；若位于边界外部，则存在侧翻风险。

---

## 8. LTR 方法与能量-相平面方法的对比

| 对比维度 | LTR 静态判断方法 | 能量-相平面动态判断方法 |
|---|---|---|
| 判断对象 | 当前左右轮载转移程度 | 当前侧倾状态是否具有越过侧翻边界的能量 |
| 主要变量 | \(F_z,a_y\) | \(\phi,\dot\phi,I_\phi,K_\phi,C_\phi,a_y\) |
| 判断边界 | \(|LTR|=1\) | \(E(\phi,\dot\phi)=E_{crit}\) |
| 是否考虑侧倾角速度 | 间接或不足 | 直接考虑 |
| 是否具有预测性 | 较弱，偏瞬时 | 较强，可反映未来侧翻趋势 |
| 适用场景 | 稳态转弯、准静态侧翻判断 | 紧急避障、双移线、高速换道、重型车辆瞬态侧翻判断 |
| 优点 | 简单、直观、实时性强 | 动态性强、能区分“当前未离地但即将侧翻”的状态 |
| 局限 | 可能滞后或保守 | 需要侧倾模型和参数标定 |

核心区别可概括为：

\[
\boxed{
LTR\ 判断的是“当前轮胎是否接近离地”；
能量-相平面法判断的是“当前侧倾状态是否具有足够能量发展为侧翻”。
}
\]

---

## 9. 横摆与侧倾综合稳定性判断

对于高速重型车辆，仅判断横摆稳定性或侧倾稳定性都不充分。更完整的稳定性判断应同时考虑：

\[
\beta,
\qquad
r,
\qquad
LTR,
\qquad
\phi,
\qquad
\dot\phi
\]

可构造综合稳定性条件：

\[
\boxed{
\begin{cases}
(\beta,r)\in\Omega_y,\\
|LTR|<LTR_{max},\\
E(\phi,\dot\phi)<\eta E_{crit}
\end{cases}
}
\]

其中：

- \(\Omega_y\)：横摆稳定域；
- \(LTR_{max}\)：LTR 工程安全阈值；
- \(\eta\)：能量安全系数，通常 \(0<\eta<1\)。

也可以定义综合风险指标：

\[
J_{stab}
=w_1\frac{|\beta|}{\beta_{max}}
+w_2\frac{|r|}{r_{max}}
+w_3\frac{|LTR|}{LTR_{max}}
+w_4\frac{E(\phi,\dot\phi)}{E_{crit}}
\]

当：

\[
J_{stab}<1
\]

可认为车辆处于综合稳定区域；当 \(J_{stab}\ge 1\) 时，说明横摆或侧倾风险已经较高。

---

## 10. 可用于论文或报告中的总结表述

基于上述推导，可以将方法总结为如下形式：

> 首先建立包含车身纵向、横向、横摆、侧倾以及四轮旋转运动的8自由度车辆动力学模型，并结合轮胎侧偏角、滑移率、非线性轮胎力和垂向载荷转移计算车辆在极限工况下的运动响应。在横摆稳定性判断中，选取质心侧偏角 \(\beta\) 和横摆角速度 \(r\) 构造 \(\beta-r\) 相平面，通过求解固定车速、前轮转角和路面附着条件下的平衡点及其吸引域，确定横摆稳定边界。在侧倾稳定性判断中，一方面采用载荷转移率 LTR 描述左右轮垂向载荷转移程度，并以 \(|LTR|=1\) 作为轮胎离地的静态侧翻临界条件；另一方面，进一步构建 \(\phi-\dot\phi\) 侧倾相平面和侧倾能量函数，以 \(E(\phi,\dot\phi)=E_{crit}\) 作为动态侧翻边界，从而同时考虑侧倾角和侧倾角速度对侧翻风险的影响。该方法相比单纯 LTR 判断具有更强的动态预测能力，尤其适用于高速换道、紧急避障和重型车辆等瞬态侧翻风险评估场景。

---

## 11. 计算流程汇总

完整计算流程如下：

1. **输入车辆参数**：\(m,I_z,I_\phi,l_f,l_r,T,h,K_\phi,C_\phi,R_w,I_w\)；
2. **输入工况参数**：\(v_x,\delta_f,\mu,T_i\)；
3. **计算轮胎运动学量**：\(\alpha_i,\lambda_i\)；
4. **计算轮胎力**：\(F_{x,i},F_{y,i}\)；
5. **计算垂向载荷**：\(F_{z,fl},F_{z,fr},F_{z,rl},F_{z,rr}\)；
6. **积分8自由度动力学模型**；
7. **横摆稳定性判断**：
   \[
   (\beta,r)\in\Omega_y
   \]
8. **侧倾静态稳定性判断**：
   \[
   |LTR|<LTR_{max}
   \]
9. **侧倾动态稳定性判断**：
   \[
   E(\phi,\dot\phi)<\eta E_{crit}
   \]
10. **输出稳定性状态**：稳定、横摆失稳风险、侧倾失稳风险或综合失稳风险。

---

## 12. 关键公式汇总

### 8自由度状态变量

\[
\mathbf{x}=\left[
 v_x,\ v_y,\ r,\ \phi,\ \dot\phi,\ \omega_{fl},\omega_{fr},\omega_{rl},\omega_{rr}
\right]^T
\]

### 质心侧偏角

\[
\beta=\arctan\left(\frac{v_y}{v_x}\right)
\]

### 横向加速度

\[
a_y=\dot v_y+v_xr
\]

### 横摆相平面

\[
\begin{cases}
\dot\beta=f_1(\beta,r;v_x,\delta_f,\mu)\\
\dot r=f_2(\beta,r;v_x,\delta_f,\mu)
\end{cases}
\]

### 横摆稳定边界

\[
\partial\Omega_y(\mathbf{x}_{ys})
\]

### LTR

\[
LTR=
\frac{(F_{z,fr}+F_{z,rr})-(F_{z,fl}+F_{z,rl})}
{F_{z,fr}+F_{z,rr}+F_{z,fl}+F_{z,rl}}
\]

### 静态 LTR 近似

\[
LTR=\frac{2h}{T}\frac{a_y}{g}
\]

### 静态侧翻边界

\[
|LTR|=1
\]

或：

\[
|a_y|=g\frac{T}{2h}
\]

### 侧倾动力学

\[
I_\phi\ddot\phi
+C_\phi\dot\phi
+K_{\phi,eff}\phi
=m_s h_s a_y
\]

### 侧倾能量函数

\[
E(\phi,\dot\phi)
=
\frac{1}{2}I_\phi\dot\phi^2
+\frac{1}{2}K_{\phi,eff}\phi^2
-m_s h_s a_y\phi
\]

### 动态侧翻边界

\[
E(\phi,\dot\phi)=E_{crit}
\]

### 动态侧倾稳定裕度

\[
S_{roll}=1-\frac{E(\phi,\dot\phi)}{E_{crit}}
\]

---

# 13. 基于 8DOF 车辆模型的鞍点稳定流形法计算相平面稳定边界

前文已经分别给出了横摆稳定性与侧倾稳定性的相平面判断方法。需要进一步说明的是：如果车辆模型采用 8DOF 形式，则系统真实状态空间是高维的，稳定边界严格来说不是二维相平面中的一条曲线，而是高维状态空间中的吸引域边界。

因此，在 8DOF 车辆模型中计算横摆或侧倾相平面稳定边界时，不能直接把完整 8DOF 系统等同于二维相平面系统，而应采用如下思路：

\[
\boxed{
\text{8DOF 车辆模型}
\rightarrow
\text{二维约化动力学或二维状态截面}
\rightarrow
\text{平衡点与鞍点识别}
\rightarrow
\text{鞍点稳定流形反向积分}
\rightarrow
\text{相平面稳定边界}
}
\]

更严谨地说：基于 8DOF 车辆动力学模型，首先在给定工况下构造横摆和侧倾相平面的二维约化动力学，或定义完整 8DOF 状态空间中的二维截面。随后求解对应约化系统的平衡点并进行线性化分类，识别稳定平衡点和鞍型临界平衡点。对于鞍点，沿其稳定特征向量两侧施加微小扰动，并对约化系统进行反向积分，得到鞍点稳定流形。该稳定流形在相平面中的投影或切片即为横摆/侧倾稳定边界。

---

## 13.1 完整 8DOF 系统与二维相平面的关系

设 8DOF 车辆模型写为：

\[
\dot{X}=F(X;p)
\]

其中：

\[
X\in\mathbb{R}^8
\]

为完整车辆状态，\(p\) 为固定工况参数，例如：

\[
p=(v_x,\delta_f,\mu,T_b,T_d,\text{vehicle parameters})
\]

如果将纵向车速 \(v_x\) 视为固定工况参数，则一种常用状态可写为：

\[
X=
\begin{bmatrix}
v_y & r & \omega_{fl} & \omega_{fr} & \omega_{rl} & \omega_{rr} & \phi & \dot\phi
\end{bmatrix}^T
\]

其中：

- \(v_y\)：质心横向速度；
- \(r\)：横摆角速度；
- \(\omega_{fl},\omega_{fr},\omega_{rl},\omega_{rr}\)：四轮角速度；
- \(\phi\)：车身侧倾角；
- \(\dot\phi\)：车身侧倾角速度。

完整系统的稳定吸引域为：

\[
\Omega(X_s)\subset\mathbb{R}^8
\]

其边界为：

\[
\partial\Omega(X_s)\subset\mathbb{R}^8
\]

在一般情况下，该边界是高维超曲面，而不是二维相平面中的曲线。因此，若希望得到横摆相平面：

\[
(\beta,r)
\]

或侧倾相平面：

\[
(\phi,\dot\phi)
\]

中的稳定边界，需要先定义二维约化系统或二维截面。

---

## 13.2 二维约化系统与二维截面

对于横摆相平面，定义：

\[
z_y=
\begin{bmatrix}
\beta\\
r
\end{bmatrix}
\]

其中：

\[
\beta=\arctan\frac{v_y}{v_x}
\]

若 \(v_x=\bar v_x\) 固定，则：

\[
v_y=\bar v_x\tan\beta
\]

因此，横摆相平面中的一个点 \((\beta,r)\) 可以嵌入完整 8DOF 状态空间：

\[
X=\Psi_y(\beta,r)
\]

例如可取：

\[
\Psi_y(\beta,r)=
\begin{bmatrix}
\bar v_x\tan\beta\\
r\\
\bar v_x/R_w\\
\bar v_x/R_w\\
\bar v_x/R_w\\
\bar v_x/R_w\\
\phi^{ref}\\
0
\end{bmatrix}
\]

其中 \(R_w\) 为车轮半径，\(\phi^{ref}\) 可取 0，也可取该工况下的准静态侧倾平衡值。

对于侧倾相平面，定义：

\[
z_r=
\begin{bmatrix}
\phi\\
\dot\phi
\end{bmatrix}
\]

同样需要将侧倾相平面点嵌入完整状态空间：

\[
X=\Psi_r(\phi,\dot\phi)
\]

例如可取：

\[
\Psi_r(\phi,\dot\phi)=
\begin{bmatrix}
v_{y,s}\\
r_s\\
\bar v_x/R_w\\
\bar v_x/R_w\\
\bar v_x/R_w\\
\bar v_x/R_w\\
\phi\\
\dot\phi
\end{bmatrix}
\]

其中 \(v_{y,s},r_s\) 为该工况下的稳态横摆运动状态。

因此，相平面边界本质上是完整高维稳定边界在二维截面上的切片：

\[
\Gamma_y=\partial\Omega(X_s)\cap \mathcal{S}_y
\]

\[
\Gamma_r=\partial\Omega(X_s)\cap \mathcal{S}_r
\]

其中：

\[
\mathcal{S}_y=\{X=\Psi_y(\beta,r)\}
\]

\[
\mathcal{S}_r=\{X=\Psi_r(\phi,\dot\phi)\}
\]

实际计算中，直接求高维边界与二维截面的交集很困难，因此通常构造二维约化动力学，再在二维约化系统中使用鞍点稳定流形法。

---

# 14. 横摆稳定边界的鞍点稳定流形法

## 14.1 横摆二维约化动力学

在固定工况：

\[
p=(\bar v_x,\delta_f,\mu,\cdots)
\]

下，选取横摆相平面状态：

\[
z_y=
\begin{bmatrix}
\beta\\
r
\end{bmatrix}
\]

由 8DOF 车辆模型计算横摆相关动力学。质心侧偏角满足：

\[
\beta=\arctan\frac{v_y}{\bar v_x}
\]

因此：

\[
\dot\beta=
\frac{\bar v_x\dot v_y}{\bar v_x^2+v_y^2}
\]

当 \(\beta\) 较小时，可近似为：

\[
\dot\beta\approx \frac{\dot v_y}{\bar v_x}
\]

车辆横向动力学为：

\[
m(\dot v_y+\bar v_x r)=\sum_i F_{y,i}
\]

因此：

\[
\dot v_y=\frac{1}{m}\sum_i F_{y,i}-\bar v_x r
\]

则横摆相平面中的 \(\dot\beta\) 可写为：

\[
\dot\beta
=
\frac{\bar v_x}{\bar v_x^2+v_y^2}
\left(
\frac{1}{m}\sum_i F_{y,i}-\bar v_x r
\right)
\]

小角度条件下：

\[
\dot\beta
\approx
\frac{1}{m\bar v_x}\sum_i F_{y,i}-r
\]

横摆角速度动力学为：

\[
\dot r=
\frac{1}{I_z}
\left[
l_f(F_{y,fl}+F_{y,fr})
-
l_r(F_{y,rl}+F_{y,rr})
+
\frac{t_w}{2}
\left(
F_{x,fr}-F_{x,fl}+F_{x,rr}-F_{x,rl}
\right)
\right]
\]

因此可得到横摆二维约化系统：

\[
\dot z_y=f_y^{red}(z_y;p)
\]

即：

\[
\begin{bmatrix}
\dot\beta\\
\dot r
\end{bmatrix}
=
\begin{bmatrix}
f_{y1}^{red}(\beta,r;p)\\
f_{y2}^{red}(\beta,r;p)
\end{bmatrix}
\]

这里的 \(F_{x,i},F_{y,i},F_{z,i}\) 仍由 8DOF 模型中的轮胎模型、载荷转移模型和侧倾准稳态关系给出。

---

## 14.2 横摆平衡点求解

横摆平衡点满足：

\[
f_y^{red}(z_{y,e};p)=0
\]

即：

\[
\dot\beta=0
\]

\[
\dot r=0
\]

求解得到一组平衡点：

\[
z_{y,e}^{(1)},z_{y,e}^{(2)},\cdots,z_{y,e}^{(n)}
\]

其中可能包括稳定横摆平衡点：

\[
z_{y,s}=(\beta_s,r_s)^T
\]

以及鞍型横摆临界平衡点：

\[
z_{y,u}=(\beta_u,r_u)^T
\]

---

## 14.3 横摆平衡点线性化与分类

对二维约化横摆系统在平衡点处线性化：

\[
\delta\dot z_y=A_y\delta z_y
\]

其中：

\[
A_y=
\left.
\frac{\partial f_y^{red}}{\partial z_y}
\right|_{z_y=z_{y,e}}
=
\begin{bmatrix}
\frac{\partial \dot\beta}{\partial \beta} &
\frac{\partial \dot\beta}{\partial r}\\
\frac{\partial \dot r}{\partial \beta} &
\frac{\partial \dot r}{\partial r}
\end{bmatrix}_{z_y=z_{y,e}}
\]

求解特征值：

\[
\lambda_1,\lambda_2
\]

若：

\[
\operatorname{Re}(\lambda_1)<0,
\quad
\operatorname{Re}(\lambda_2)<0
\]

则该平衡点为稳定横摆平衡点。

若：

\[
\operatorname{Re}(\lambda_1)<0,
\quad
\operatorname{Re}(\lambda_2)>0
\]

或反之，则该平衡点为鞍点。该鞍点对应横摆相平面中稳定域与失稳域之间的临界结构。

---

## 14.4 横摆鞍点稳定流形反向积分

设横摆鞍点为：

\[
z_{y,u}
\]

其稳定特征值为：

\[
\lambda_s
\]

对应稳定特征向量为：

\[
v_{y,s}
\]

在鞍点两侧施加微小扰动：

\[
z_{y,0}^{+}=z_{y,u}+\epsilon v_{y,s}
\]

\[
z_{y,0}^{-}=z_{y,u}-\epsilon v_{y,s}
\]

其中 \(\epsilon>0\) 为小扰动量。

为了展开稳定流形，需要对原系统进行反向时间积分：

\[
\dot z_y=-f_y^{red}(z_y;p)
\]

从 \(z_{y,0}^{+}\) 和 \(z_{y,0}^{-}\) 分别积分，可得到两条稳定流形分支：

\[
\Gamma_y^+
\]

\[
\Gamma_y^-
\]

横摆相平面稳定边界为：

\[
\boxed{
\partial\Omega_y
=
\Gamma_y^+\cup\Gamma_y^-
}
\]

该边界将横摆相平面划分为两个区域：

\[
\Omega_y=
\{(\beta,r):\text{轨迹最终回到稳定横摆平衡点}\}
\]

以及横摆失稳区域。

---

## 14.5 横摆边界的物理含义

横摆鞍点稳定流形边界表示：

\[
\boxed{
\text{车辆横摆状态由可恢复区域进入不可恢复区域的临界分界线}
}
\]

如果当前状态 \((\beta,r)\) 位于稳定边界内部，则车辆横摆运动会收敛到稳态转弯平衡点；如果位于边界外部，则可能出现侧滑、甩尾、横摆发散或进入其他失稳状态。

需要注意的是，这一边界依赖于固定工况：

\[
(v_x,\delta_f,\mu)
\]

因此不同车速、前轮转角和路面附着系数下，横摆稳定边界不同。

---

# 15. 侧倾稳定边界的鞍点稳定流形法

## 15.1 侧倾二维约化动力学

侧倾相平面选取：

\[
z_r=
\begin{bmatrix}
\phi\\
\dot\phi
\end{bmatrix}
\]

令：

\[
\omega_\phi=\dot\phi
\]

则：

\[
z_r=
\begin{bmatrix}
\phi\\
\omega_\phi
\end{bmatrix}
\]

从 8DOF 车辆模型中提取侧倾动力学，可写为：

\[
\dot\phi=\omega_\phi
\]

\[
I_\phi\dot\omega_\phi
=
M_{roll}(\phi,\omega_\phi,a_y)
-
C_\phi\omega_\phi
-
K_\phi\phi
+
M_g(\phi)
\]

一种常用非线性侧倾模型为：

\[
I_\phi\ddot\phi
+
C_\phi\dot\phi
+
K_\phi\phi
-
m_sgh_s\sin\phi
=
m_sh_sa_y\cos\phi
\]

因此：

\[
\dot z_r=f_r^{red}(z_r;p)
\]

即：

\[
\begin{bmatrix}
\dot\phi\\
\dot\omega_\phi
\end{bmatrix}
=
\begin{bmatrix}
\omega_\phi\\
\frac{1}{I_\phi}
\left(
m_sh_sa_y\cos\phi
+
m_sgh_s\sin\phi
-
C_\phi\omega_\phi
-
K_\phi\phi
\right)
\end{bmatrix}
\]

其中 \(a_y\) 为横向加速度。为了得到二维自治侧倾系统，通常需要在给定工况下固定或准静态给定：

\[
a_y=\bar a_y
\]

例如可取稳态转弯横向加速度：

\[
\bar a_y=\bar v_x r_s
\]

若不固定 \(a_y\)，而使其随 \(\beta,r\) 变化，则系统应扩展为横摆-侧倾耦合四维系统：

\[
z=
\begin{bmatrix}
\beta&r&\phi&\dot\phi
\end{bmatrix}^T
\]

---

## 15.2 侧倾平衡点求解

侧倾平衡点满足：

\[
f_r^{red}(z_{r,e};p)=0
\]

即：

\[
\dot\phi=0
\]

\[
\ddot\phi=0
\]

因此：

\[
\omega_\phi=0
\]

并且：

\[
m_sh_s\bar a_y\cos\phi_e
+
m_sgh_s\sin\phi_e
-
K_\phi\phi_e
=0
\]

记：

\[
g_r(\phi_e)=
m_sh_s\bar a_y\cos\phi_e
+
m_sgh_s\sin\phi_e
-
K_\phi\phi_e
\]

则侧倾平衡点由下式确定：

\[
g_r(\phi_e)=0
\]

可能存在多个平衡点：

\[
z_{r,e}^{(1)},z_{r,e}^{(2)},\cdots,z_{r,e}^{(n)}
\]

其中小侧倾角附近的平衡点通常为稳定侧倾平衡点，靠近侧翻临界位置的平衡点可能为鞍型临界平衡点。

---

## 15.3 侧倾平衡点线性化与分类

侧倾二维系统为：

\[
\dot z_r=
\begin{bmatrix}
\omega_\phi\\
f_{r2}(\phi,\omega_\phi)
\end{bmatrix}
\]

在平衡点 \((\phi_e,0)\) 处线性化：

\[
\delta\dot z_r=A_r\delta z_r
\]

其中：

\[
A_r=
\begin{bmatrix}
0 & 1\\
\frac{\partial f_{r2}}{\partial \phi} &
\frac{\partial f_{r2}}{\partial \omega_\phi}
\end{bmatrix}_{(\phi_e,0)}
\]

对上述非线性侧倾模型，有：

\[
\frac{\partial f_{r2}}{\partial \omega_\phi}
=
-
\frac{C_\phi}{I_\phi}
\]

\[
\frac{\partial f_{r2}}{\partial \phi}
=
\frac{1}{I_\phi}
\left(
-m_sh_s\bar a_y\sin\phi_e
+
m_sgh_s\cos\phi_e
-
K_\phi
\right)
\]

求解 \(A_r\) 的特征值：

\[
\lambda_1,\lambda_2
\]

若：

\[
\operatorname{Re}(\lambda_1)<0,
\quad
\operatorname{Re}(\lambda_2)<0
\]

则该点为稳定侧倾平衡点。

若：

\[
\operatorname{Re}(\lambda_1)<0,
\quad
\operatorname{Re}(\lambda_2)>0
\]

或反之，则该点为侧倾鞍点。该鞍点对应动态侧翻边界上的临界平衡点。

---

## 15.4 侧倾鞍点稳定流形反向积分

设侧倾鞍点为：

\[
z_{r,u}=
\begin{bmatrix}
\phi_u\\
0
\end{bmatrix}
\]

其稳定特征向量为：

\[
v_{r,s}
\]

在鞍点两侧施加微小扰动：

\[
z_{r,0}^{+}=z_{r,u}+\epsilon v_{r,s}
\]

\[
z_{r,0}^{-}=z_{r,u}-\epsilon v_{r,s}
\]

对侧倾二维约化系统进行反向积分：

\[
\dot z_r=-f_r^{red}(z_r;p)
\]

得到两条侧倾稳定流形分支：

\[
\Gamma_r^+
\]

\[
\Gamma_r^-
\]

因此侧倾相平面中的动态稳定边界为：

\[
\boxed{
\partial\Omega_r
=
\Gamma_r^+\cup\Gamma_r^-
}
\]

该边界将侧倾相平面划分为可恢复区域与动态侧翻风险区域。

---

## 15.5 侧倾边界与能量边界的关系

对于无阻尼或弱阻尼的侧倾系统，能量法得到的边界：

\[
E(\phi,\dot\phi)=E_c
\]

可以看作经过临界鞍点的分界轨道。在理想保守系统中，该等能量曲线与鞍点稳定流形重合。

若系统存在阻尼：

\[
C_\phi>0
\]

则总能量会随时间耗散，此时真实动态边界更严格地由鞍点稳定流形确定，而能量边界通常是近似边界或保守边界。

因此，在侧倾稳定性分析中：

\[
\boxed{
\text{能量法给出物理直观的动态侧翻边界，鞍点稳定流形法给出更严格的相平面分界结构。}
}
\]

---

# 16. 横摆-侧倾耦合情况下的四维约化系统

如果不希望完全分离横摆和侧倾，可以构造横摆-侧倾耦合四维约化系统：

\[
z=
\begin{bmatrix}
\beta\\
r\\
\phi\\
\dot\phi
\end{bmatrix}
\]

其动力学为：

\[
\dot z=f^{red}(z;p)
\]

其中横摆方程中的轮胎力受侧倾载荷转移影响，侧倾方程中的横向加速度又由横摆状态决定。因此二者存在双向耦合：

\[
F_{z,i}=F_{z,i}(\phi,\dot\phi,a_y)
\]

\[
a_y=a_y(\beta,r,\dot\beta)
\]

在四维系统中，稳定域边界一般是三维流形。若要求横摆相平面边界，可取切片：

\[
\phi=\phi_s,
\quad
\dot\phi=0
\]

得到 \((\beta,r)\) 平面上的边界。

若要求侧倾相平面边界，可取切片：

\[
\beta=\beta_s,
\quad
r=r_s
\]

得到 \((\phi,\dot\phi)\) 平面上的边界。

这种方法比单独二维约化更严谨，但计算复杂度明显更高。

---

# 17. 实际计算流程总结

基于 8DOF 车辆模型采用鞍点稳定流形法计算横摆/侧倾相平面稳定边界时，可按如下流程执行。

## 17.1 横摆稳定边界计算流程

\[
\boxed{
\begin{aligned}
&\text{1. 固定工况 }p=(v_x,\delta_f,\mu,\cdots)\\
&\text{2. 选取横摆相平面 }z_y=(\beta,r)^T\\
&\text{3. 定义 8DOF 到横摆相平面的约化映射 }X=\Psi_y(\beta,r)\\
&\text{4. 由 8DOF 模型计算 }\dot\beta,\dot r\\
&\text{5. 得到二维约化横摆系统 }\dot z_y=f_y^{red}(z_y;p)\\
&\text{6. 求解平衡点 }f_y^{red}(z_y;p)=0\\
&\text{7. 对平衡点线性化并识别稳定点和鞍点}\\
&\text{8. 沿鞍点稳定特征向量两侧施加微小扰动}\\
&\text{9. 对 }\dot z_y=-f_y^{red}(z_y;p)\text{ 反向积分}\\
&\text{10. 得到横摆相平面稳定边界}
\end{aligned}
}
\]

## 17.2 侧倾稳定边界计算流程

\[
\boxed{
\begin{aligned}
&\text{1. 固定工况 }p=(v_x,\delta_f,\mu,\bar a_y,\cdots)\\
&\text{2. 选取侧倾相平面 }z_r=(\phi,\dot\phi)^T\\
&\text{3. 定义 8DOF 到侧倾相平面的约化映射 }X=\Psi_r(\phi,\dot\phi)\\
&\text{4. 由 8DOF 模型提取侧倾动力学}\\
&\text{5. 得到二维约化侧倾系统 }\dot z_r=f_r^{red}(z_r;p)\\
&\text{6. 求解侧倾平衡点 }f_r^{red}(z_r;p)=0\\
&\text{7. 对平衡点线性化并识别稳定侧倾点和临界侧翻鞍点}\\
&\text{8. 沿鞍点稳定特征向量两侧施加微小扰动}\\
&\text{9. 对 }\dot z_r=-f_r^{red}(z_r;p)\text{ 反向积分}\\
&\text{10. 得到侧倾相平面稳定边界}
\end{aligned}
}
\]

---

# 18. 方法适用性说明

该方法的关键前提是：在给定工况下，约化系统中存在稳定平衡点和鞍型临界平衡点。如果不存在稳定平衡点，则不能定义围绕该稳定平衡点的稳定边界；如果不存在鞍点，则稳定边界可能由其他结构决定，例如不稳定极限环、非光滑切换边界、危险集合边界或系统约束边界。

因此，8DOF 系统中的鞍点稳定流形法应理解为：

\[
\boxed{
\text{基于 8DOF 车辆动力学信息构造低维相平面约化模型，并在该约化模型中追踪鞍点稳定流形。}
}
\]

它不是直接在完整 8DOF 状态空间中求一条二维边界曲线，而是通过约化、截面或投影得到横摆/侧倾相平面中的稳定边界。

