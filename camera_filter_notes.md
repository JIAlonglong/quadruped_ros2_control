# D435i 相机滤波配置笔记

## 1. 相机节点启动命令

**一行命令（直接复制）：**

```bash
ros2 run realsense2_camera realsense2_camera_node --ros-args -r __ns:=/rgbd_d435 -p enable_color:=true -p enable_depth:=true -p filters:=spatial,temporal,holes_filling -r color/image_raw:=image -r depth/image_rect_raw:=depth_image
```

多行写法（同上，便于阅读）：

```bash
ros2 run realsense2_camera realsense2_camera_node \
  --ros-args \
    -r __ns:=/rgbd_d435 \
    -p enable_color:=true \
    -p enable_depth:=true \
    -p filters:=spatial,temporal,holes_filling \
    -r color/image_raw:=image \
    -r depth/image_rect_raw:=depth_image
```

运行后主要话题：

- 彩色图：`/rgbd_d435/camera/image`
- 深度图（已滤波）：`/rgbd_d435/camera/depth_image`

`StateRL` 控制器订阅 `/rgbd_d435/camera/depth_image`，并发布可视化用的 `/depth_image/processed`。

## 2. realsense2_camera 滤波参数

节点名：`/rgbd_d435/camera`

### 2.1 打开滤波器

```bash
ros2 param set /rgbd_d435/camera spatial_filter.enable          true
ros2 param set /rgbd_d435/camera temporal_filter.enable         true
ros2 param set /rgbd_d435/camera hole_filling_filter.enable     true
```

### 2.2 空间滤波 spatial_filter

```bash
ros2 param set /rgbd_d435/camera spatial_filter.filter_magnitude     5
ros2 param set /rgbd_d435/camera spatial_filter.filter_smooth_alpha  0.7
ros2 param set /rgbd_d435/camera spatial_filter.filter_smooth_delta  5.0
ros2 param set /rgbd_d435/camera spatial_filter.holes_fill           5
```

含义（直观）：

- `filter_magnitude`：迭代次数，数值越大空间平滑越强；
- `filter_smooth_alpha`：空间 EMA 系数，0 接近无限滤波，1 表示关闭滤波；
- `filter_smooth_delta`：保边阈值，越小越保边，当前设置偏向多平滑；
- `holes_fill`：在空间滤波阶段对局部 0 深度做补洞，5 较为激进。

### 2.3 时间滤波 temporal_filter

```bash
ros2 param set /rgbd_d435/camera temporal_filter.filter_smooth_alpha 0.7
ros2 param set /rgbd_d435/camera temporal_filter.filter_smooth_delta 5.0
ros2 param set /rgbd_d435/camera temporal_filter.holes_fill          5
```

- 在时间维度对每个像素做 EMA 平滑；
- `holes_fill` 使用历史帧补短时间出现的 0 深度，减少闪烁的“洞”。

### 2.4 独立 hole_filling_filter

```bash
ros2 param set /rgbd_d435/camera hole_filling_filter.holes_fill 2
```

额外基于邻域的补洞，进一步减少稀疏 0 像素。

### 2.5 查看当前参数

```bash
ros2 param list /rgbd_d435/camera | grep -E "spatial|temporal|hole|filter"
ros2 param dump /rgbd_d435/camera
```

## 3. 控制端深度处理（StateRL）

### 3.1 订阅与可视化

- 订阅滤波后的深度：

  ```cpp
  depth_image_sub_ = node_->create_subscription<sensor_msgs::msg::Image>(
      "/rgbd_d435/camera/depth_image", 10, ...);
  ```

- 回调中：
  - 调用 `preprocessDepthImage(msg)` 生成 RL 输入 `obs_.depth_latent`（尺寸 `1×58×87`，范围约 `[-0.5, 0.5]`）；
  - 对深度做一次 5×5 中值滤波，并发布到 `/depth_image/processed` 供 RViz 查看。

### 3.2 RL 观测预处理概述

`preprocessDepthImage` 的主要步骤：

1. 支持 `16UC1` / `32FC1`，统一转为米单位的 `cv::Mat`；
2. 利用 `params_.max_depth` 截断异常大值，填补 NaN；
3. 裁剪边缘区域，对齐 Extreme‑Parkour 使用的 ROI；
4. 使用双三次插值缩放到 `58×87`；
5. 截断到 `[0, max_depth]`，再除以 `max_depth` 归一化到 `[0,1]`，然后整体减 `0.5`；
6. 转为 `torch::Tensor`，形状 `[1, 58, 87]`，作为策略网络的深度观测。

