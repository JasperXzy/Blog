---
title: "在 RDK X5 上部署 YOLO11：基于 ROS2 的实时障碍物检测"
pubDatetime: 2025-06-29T19:37:12+08:00
featured: false
draft: false
category: Projects
description: "在 RDK X5 上部署 YOLO11，完成 NV12 预处理、BPU 推理、DFL 解码、NMS 与 ROS2 感知消息发布"
---

这篇文章记录如何在 RDK X5 上部署经过量化的 YOLO11 检测模型，并将推理流程封装为 ROS2 节点。完整链路从摄像头的 NV12 图像开始，经过 Letterbox、BPU 推理、DFL 解码和 NMS，最终发布 `ai_msgs::msg::PerceptionTargets` 消息，供车辆控制节点直接订阅。

本文对应的代码位于 [racing_obstacle_detection](https://github.com/JasperXzy/racing_obstacle_detection)。当前配置检测两类目标：

- `construction_cone`：锥桶
- `parking_sign`：停车标志

仓库记录的实际消息输出约为 26–30 FPS。二维码识别属于另一条感知链路，不在这个 YOLO11 节点的类别列表中。

## 系统数据流

整个节点可以概括为以下数据通路：

```text
/hbmem_img
    │  640 × 480 NV12
    ▼
NV12 Letterbox
    │  640 × 640 NV12
    ▼
YOLO11 .bin 模型 / RDK X5 BPU
    │  3 组分类张量 + 3 组回归张量
    ▼
阈值筛选 → DFL 解码 → dist2bbox → 分类别 NMS
    │
    ▼
逆 Letterbox 坐标映射
    │
    ▼
/racing_obstacle_detection
    ai_msgs::msg::PerceptionTargets
```

这里有三个实现重点：

1. 摄像头与模型都使用 NV12，避免在主链路中反复转换为 BGR。
2. YOLO11 的三个检测尺度分别处理小、中、大目标。
3. 发布前必须撤销 Letterbox 的缩放和填充，否则边界框无法与原图对齐。

## 运行环境与 ROS2 接口

项目依赖 RDK X5 的 BPU 推理库、ROS2/TROS、OpenCV、`hbm_img_msgs`、`ai_msgs` 和 `nlohmann_json`。

| 方向   | 话题                         | 消息类型                          | 作用                       |
| ------ | ---------------------------- | --------------------------------- | -------------------------- |
| 输入   | `/hbmem_img`                 | `hbm_img_msgs::msg::HbmMsg1080P`  | 接收摄像头发布的 NV12 图像 |
| 输出   | `/racing_obstacle_detection` | `ai_msgs::msg::PerceptionTargets` | 发布类别、置信度和矩形框   |
| 可视化 | `/image`                     | MJPEG 图像流                      | 供 WebSocket 节点显示      |

图像订阅使用 `rclcpp::SensorDataQoS()`。这种 QoS 更适合实时传感器数据：当处理速度暂时跟不上输入时，系统优先处理新帧，而不是积压大量过期画面。

## 准备 YOLO11 模型与配置

RDK X5 运行的不是 PyTorch `.pt` 文件，而是经过工具链编译的 `.bin` 模型。本项目期望模型具备以下接口：

- 单个 NV12 输入
- NCHW 布局
- 640 × 640 输入尺寸
- 6 个输出张量
- 三个分类输出与三个边界框回归输出
- 回归张量使用 SCALE 量化信息

项目中的 `config/yolo11.json` 如下：

```json
{
  "model_file": "config/yolov11n.bin",
  "class_num": 2,
  "dnn_Parser": "yolov11",
  "cls_names_list": ["construction_cone", "parking_sign"],
  "preprocess_type": 1,
  "nms_threshold": 0.5,
  "score_threshold": 0.25,
  "nms_top_k": 300,
  "reg": 16,
  "font_size": 1.0,
  "font_thickness": 1.0,
  "line_size": 2.0
}
```

其中最影响结果的参数是：

- `score_threshold`：候选框置信度阈值
- `nms_threshold`：NMS 的 IoU 阈值
- `nms_top_k`：进入 NMS 的候选框上限
- `reg`：DFL 每个方向的离散区间数量
- `cls_names_list`：模型输出类别与业务名称的映射

## 编译与启动

将仓库作为一个 ROS2 工作空间使用：

```bash
git clone https://github.com/JasperXzy/racing_obstacle_detection.git
cd racing_obstacle_detection

colcon build --symlink-install
source install/setup.bash
```

启动摄像头：

```bash
ros2 launch origincar_bringup camera.launch.py
```

在另一个终端启动检测节点：

```bash
cd racing_obstacle_detection
source install/setup.bash
ros2 launch racing_obstacle_detection obstacle_detection.launch.py
```

如果还需要 Web 可视化，可以改用：

```bash
ros2 launch racing_obstacle_detection obstacle_detection_web.launch.py
```

最后检查输出话题与板卡负载：

```bash
ros2 topic echo /racing_obstacle_detection
sudo hrut_somstatus
```

## ROS2 回调如何串起推理链路

节点订阅 `/hbmem_img` 后，在同一个回调中完成预处理、推理、后处理和消息发布：

```cpp
void img_callback(
    const hbm_img_msgs::msg::HbmMsg1080P::SharedPtr msg) {
    const auto input_timestamp = msg->time_stamp;
    auto &frame = msg->data;

    constexpr int src_w = 640;
    constexpr int src_h = 480;
    constexpr int dst_w = 640;
    constexpr int dst_h = 640;

    std::vector<uint8_t> output_nv12(dst_w * dst_h * 3 / 2);

    int x_shift = 0;
    int y_shift = 0;
    float x_scale = 1.0F;
    float y_scale = 1.0F;

    letterbox_nv12(
        frame.data(),
        src_w,
        src_h,
        output_nv12.data(),
        dst_w,
        dst_h,
        x_shift,
        y_shift,
        x_scale,
        y_scale);

    detector_.detect(output_nv12.data());
    detector_.postprocessing(
        x_shift, y_shift, x_scale, y_scale, src_w, src_h);

    publish_detection_results(input_timestamp);
}
```

这段代码刻意保留了 `x_shift`、`y_shift`、`x_scale` 和 `y_scale`。它们不仅服务于预处理，也决定了后处理能否把检测框准确映射回 640 × 480 原图。

## NV12 Letterbox 预处理

NV12 包含一个完整分辨率的 Y 平面，以及一个宽高各减半、UV 交错存储的色度平面。处理时不能把整块内存当作普通灰度图直接缩放，必须分别处理两个平面。

缩放比例取目标宽高约束中的较小值：

$$
s = \min\left(\frac{W_d}{W_s}, \frac{H_d}{H_s}\right)
$$

缩放后的图像居中放置，剩余区域填充灰度值 127：

```cpp
const uint8_t *y_in = nv12;
const uint8_t *uv_in = nv12 + src_w * src_h;

cv::Mat y_plane(src_h, src_w, CV_8UC1, const_cast<uint8_t *>(y_in));
cv::Mat uv_plane(src_h / 2, src_w / 2, CV_8UC2,
                 const_cast<uint8_t *>(uv_in));

const float scale = std::min(
    1.0F * dst_w / src_w,
    1.0F * dst_h / src_h);

const int new_w = static_cast<int>(src_w * scale + 0.5F);
const int new_h = static_cast<int>(src_h * scale + 0.5F);

x_shift = (dst_w - new_w) / 2;
y_shift = (dst_h - new_h) / 2;
x_scale = scale;
y_scale = scale;
```

Y 平面按 `new_w × new_h` 缩放；UV 平面的尺寸和偏移量都要除以 2。完成填充后，再把两个平面依次写回输出缓冲区。

> [!TIP] 注意色度对齐
> NV12 使用 4:2:0 色度采样。自定义其他输入尺寸时，应确保缩放尺寸和填充偏移满足 UV 平面对齐要求，否则容易出现色偏或边缘错位。

## 加载模型并执行 BPU 推理

模型初始化阶段应尽早验证输入输出接口。这样当模型转换方式与后处理代码不匹配时，程序可以在启动阶段直接失败，而不是产生难以解释的错误框。

```cpp
hbDNNInitializeFromFiles(
    &packed_dnn_handle,
    &model_file_name,
    1);

hbDNNGetModelHandle(
    &dnn_handle,
    packed_dnn_handle,
    model_name);

hbDNNGetInputTensorProperties(
    &input_properties,
    dnn_handle,
    0);

hbDNNGetOutputCount(
    &output_count,
    dnn_handle);

if (input_properties.tensorType != HB_DNN_IMG_TYPE_NV12 ||
    input_properties.tensorLayout != HB_DNN_LAYOUT_NCHW ||
    output_count != 6) {
    throw std::runtime_error("Unexpected YOLO11 model interface");
}
```

每帧推理的关键步骤是：

1. 为输入和输出张量准备 BPU 可访问内存。
2. 写入 NV12 数据后执行 cache clean。
3. 调用 `hbDNNInfer` 并等待任务完成。
4. 读取输出前执行 cache invalidate。
5. 后处理结束后释放任务和内存。

```cpp
input.properties = input_properties;
hbSysAllocCachedMem(
    &input.sysMem[0],
    3 * input_H * input_W / 2);

std::memcpy(
    input.sysMem[0].virAddr,
    input_nv12,
    3 * input_H * input_W / 2);

hbSysFlushMem(
    &input.sysMem[0],
    HB_SYS_MEM_CACHE_CLEAN);

HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
hbDNNInfer(
    &task_handle,
    &output,
    &input,
    dnn_handle,
    &infer_ctrl_param);

hbDNNWaitTaskDone(task_handle, 0);
```

当前实现会在每一帧分配并释放输入输出内存，逻辑清晰但会带来额外开销。若需要长期稳定运行，可在模型初始化后预分配缓冲区，并在后续帧中重复使用。

## 三尺度输出与 DFL 解码

640 × 640 输入对应三个检测尺度：

| 步长 | 特征图尺寸 | 主要目标 |
| ---- | ---------- | -------- |
| 8    | 80 × 80    | 小目标   |
| 16   | 40 × 40    | 中目标   |
| 32   | 20 × 20    | 大目标   |

每个尺度包含一组分类张量和一组回归张量。代码先在分类 logits 上找最大类别，再使用 Sigmoid 的单调性提前筛选候选框。

若置信度阈值为 $p_{\min}$，对应的原始 logit 阈值为：

$$
z_{\min} = \log\left(\frac{p_{\min}}{1-p_{\min}}\right)
$$

这正是代码中 `-log(1 / score_threshold - 1)` 的含义。先比较 logit，可以避免对大量低分候选执行 Sigmoid、反量化和 DFL。

DFL 将每个方向的离散分布转换为连续距离。设量化原始值为 $q_j$，缩放系数为 $a_j$，离散区间数为 $r$，则距离期望为：

$$
d = \frac{\sum_{j=0}^{r-1} j\exp(q_j a_j)}
         {\sum_{j=0}^{r-1} \exp(q_j a_j)}
$$

对应的一尺度核心代码如下：

```cpp
int cls_id = 0;
for (int i = 1; i < class_num; ++i) {
    if (cur_cls_raw[i] > cur_cls_raw[cls_id]) {
        cls_id = i;
    }
}

if (cur_cls_raw[cls_id] < conf_threshold_raw) {
    cls_raw += class_num;
    bbox_raw += reg * 4;
    continue;
}

const float score =
    1.0F / (1.0F + std::exp(-cur_cls_raw[cls_id]));

float ltrb[4] = {};
for (int side = 0; side < 4; ++side) {
    float sum = 0.0F;

    for (int j = 0; j < reg; ++j) {
        const int index = reg * side + j;
        const float value = std::exp(
            static_cast<float>(cur_bbox_raw[index]) *
            bbox_scale[index]);

        ltrb[side] += value * j;
        sum += value;
    }

    ltrb[side] /= sum;
}
```

得到左、上、右、下四个距离后，根据当前网格位置和步长转换为 `xyxy` 边界框。三个尺度都处理完成后，对每个类别分别执行 NMS：

```cpp
std::vector<std::vector<int>> indices(class_num);

for (int cls_id = 0; cls_id < class_num; ++cls_id) {
    cv::dnn::NMSBoxes(
        bboxes[cls_id],
        scores[cls_id],
        score_threshold,
        nms_threshold,
        indices[cls_id],
        1.0F,
        nms_top_k);
}
```

## 撤销 Letterbox 坐标变换

模型输出的边界框位于 640 × 640 Letterbox 坐标系中。发布到 ROS2 前，需要先减去填充偏移，再除以缩放比例：

```cpp
float x = (box.x - x_shift) / x_scale;
float y = (box.y - y_shift) / y_scale;
float w = box.width / x_scale;
float h = box.height / y_scale;

x = std::clamp(x, 0.0F, static_cast<float>(src_w));
y = std::clamp(y, 0.0F, static_cast<float>(src_h));
w = std::min(w, static_cast<float>(src_w) - x);
h = std::min(h, static_cast<float>(src_h) - y);
```

如果检测框整体出现固定方向的偏移，首先应检查这里使用的缩放与偏移参数是否和预处理阶段完全一致。

## 发布 PerceptionTargets 消息

后处理结果以 `Target → Roi → RegionOfInterest` 的层级写入消息：

```cpp
ai_msgs::msg::PerceptionTargets message;
message.header.stamp = timestamp;

for (const auto &object : detector_.get_detected_objects()) {
    ai_msgs::msg::Target target;
    target.type = object.class_name;
    target.track_id = 0;

    ai_msgs::msg::Roi roi;
    roi.type = "rect";
    roi.rect.x_offset = object.x;
    roi.rect.y_offset = object.y;
    roi.rect.width = object.width;
    roi.rect.height = object.height;
    roi.rect.do_rectify = false;
    roi.confidence = object.confidence;

    target.rois.push_back(roi);
    message.targets.push_back(target);
}

publisher_->publish(message);
```

使用 `ros2 topic echo /racing_obstacle_detection` 可以看到类似输出：

```yaml
fps: 30
targets:
  - type: parking_sign
    track_id: 0
    rois:
      - rect:
          x_offset: 104
          y_offset: 42
          width: 13
          height: 12
        confidence: 0.5756
  - type: construction_cone
    track_id: 0
    rois:
      - rect:
          x_offset: 105
          y_offset: 37
          width: 12
          height: 25
        confidence: 0.5516
```

## 总结

这套实现的关键不只是“让 YOLO11 在 RDK X5 上跑起来”，而是打通一条可以直接接入机器人控制系统的感知链路：

- 输入端直接处理 NV12，减少颜色格式转换。
- 推理端验证模型接口并调用 BPU。
- 后处理端完成三尺度 DFL 解码、NMS 和逆 Letterbox。
- 系统端使用标准 ROS2 感知消息发布结果。

代码仓库：[JasperXzy/racing_obstacle_detection](https://github.com/JasperXzy/racing_obstacle_detection)
