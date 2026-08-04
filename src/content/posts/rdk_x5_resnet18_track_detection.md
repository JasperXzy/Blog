---
title: "在 RDK X5 上部署 ResNet18：基于 ROS2 的赛道中心点检测"
pubDatetime: 2025-06-13T20:00:00+08:00
featured: false
draft: false
category: Projects
description: "从数据采集、中心点标注和 ResNet18 多任务训练，到 RDK X5 BPU 推理与 ROS2 赛道中心点发布"
---

在自动驾驶小车的感知系统中，障碍物检测解决“前方有什么”，赛道检测则要回答“车辆应该朝哪里行驶”。对于结构固定、实时性要求较高的竞速赛道，与其对整幅图像做语义分割，不如直接预测前方赛道中心点，再把这个点交给下游控制器计算转向量。

本文记录 [racing_track_detection](https://github.com/JasperXzy/racing_track_detection) 项目的完整实现：采集赛道图像、使用 LabelMe 标注中心点、训练三输出 ResNet18、导出 ONNX，并在 RDK X5 上通过 ROS2 和 BPU 完成实时推理。与[在 RDK X5 上部署 YOLO11：基于 ROS2 的实时障碍物检测](/posts/rdk_x5_yolo11_detect_obstacle/)配合后，可以形成赛道跟随与障碍物感知两条相互独立的感知链路。

## 为什么选择中心点回归

赛道分割能够输出完整的可行驶区域，但同时增加了标注成本、后处理复杂度和板端计算量。这个项目只关心图像下方某个前视点对应的赛道中心，因此将任务简化为三个输出：

- $x$：赛道中心点的横坐标
- $y$：赛道中心点的纵坐标
- $c$：当前画面存在有效中线的置信度 logit

模型使用 ResNet18 提取视觉特征，最后的全连接层从 512 维特征映射到上述三个值。坐标回归提供转向依据，置信度分支则负责判断当前预测是否可信。

这种方案的优势是输出紧凑、后处理简单，代价是模型不会给出完整赛道形状，而且对中心点标注策略和摄像头安装位置更敏感。

## 系统数据流

项目包含离线训练和板端推理两条链路：

```text
离线训练

/image_raw
    │
    ▼
截取图像底部 640 × 224 区域
    │
    ▼
LabelMe 标注赛道中心点
    │
    ▼
训练集 / 测试集
    │
    ▼
ResNet18：x_norm、y_norm、confidence_logit
    │
    ▼
PyTorch .pt → ONNX → RDK X5 .bin
```

```text
板端推理

/hbmem_img：640 × 480 NV12
    │
    ▼
底部 ROI 裁剪并缩放到 224 × 224
    │
    ▼
RDK X5 BPU / ResNet18
    │
    ▼
坐标反归一化 + 置信度 Sigmoid
    │
    ▼
/racing_track_center_detection
    ai_msgs::msg::PerceptionTargets
```

节点还订阅 `/sign4return`：收到 `5` 时暂停巡线推理，收到 `6` 时恢复。这使上层状态机能够在人工接管或特殊赛段中切换控制权。

## ROS2 接口与依赖

板端节点基于 TROS 的 `dnn_node` 封装 BPU 推理，并使用 `hobot_cv` 处理 NV12 图像。

| 方向 | 话题                             | 消息类型                          | 作用                   |
| ---- | -------------------------------- | --------------------------------- | ---------------------- |
| 输入 | `/hbmem_img`                     | `hbm_img_msgs::msg::HbmMsg1080P`  | 接收 640×480 NV12 图像 |
| 输入 | `/sign4return`                   | `std_msgs::msg::Int32`            | 暂停或恢复巡线         |
| 输出 | `/racing_track_center_detection` | `ai_msgs::msg::PerceptionTargets` | 发布中心点与置信度     |

主要依赖包括：

- ROS2/TROS 与 `rclcpp`
- `dnn_node`
- `hbm_img_msgs` 与 `ai_msgs`
- OpenCV 与 `hobot_cv`
- RDK X5 BPU 运行时

## 数据采集：只保留与巡线相关的区域

`utils/image_capture.py` 订阅普通的 `sensor_msgs::msg::Image`，按照设定帧率保存两份图像：

- 完整画面保存到障碍物数据目录
- 图像底部 224 行保存到赛道数据目录

```python
self.detection_image = self.bridge.imgmsg_to_cv2(msg)
self.line_follow_image = self.detection_image[255:479, :, :].copy()

cv.imwrite(track_image_path, self.line_follow_image)
cv.imwrite(obstacle_image_path, self.detection_image)
```

以每秒 2 帧采集，并把赛道 ROI 写入 `raw_image`：

```bash
python utils/image_capture.py --ros-args \
  -p fps:=2.0 \
  -p sub_img_topic:=/image_raw \
  -p track_img_folder:=raw_image \
  -p obstacle_img_folder:=obstacle_image
```

只保留图像下方区域有两个目的：减少无关背景干扰，同时让网络把容量集中在车辆即将驶入的赛道区域。

采集脚本当前使用 `[255:479)`，而板端推理裁剪的是图像底部 `[256:480)`。两者尺寸同为 224 行，但存在 1 像素偏移。实际训练时应统一为同一个区间，避免不必要的预处理差异。

## 使用 LabelMe 标注中心点

每张图片只需要标注一个 `point`，表示当前 ROI 中期望车辆跟随的赛道中心。没有可靠中线的图片可以不添加点，用作置信度分支的负样本。

`labelme_to_resnet.py` 将图片与 LabelMe JSON 转换为以下目录：

```text
line_follow_dataset/
├── train/
│   ├── image/
│   └── label/
└── test/
    ├── image/
    └── label/
```

执行转换：

```bash
python utils/labelme_to_resnet.py \
  --source_folder_path raw_image \
  --target_folder_path line_follow_dataset
```

每个标签文件包含三个值：

```text
x y has_midline
```

存在中心点时写入 `x y 1`；没有点、JSON 缺失或标注无法解析时写入 `NaN NaN 0`。脚本默认按 8:2 随机划分训练集和测试集。

为了让实验可以复现，建议在划分前固定随机种子，或者预先生成一份训练集与测试集清单。同一段连续视频的相邻帧也不应随机分散到两个集合，否则测试结果容易偏高。

## 坐标归一化

原始 ROI 的尺寸为 640×224，模型输入则固定为 224×224。训练代码把标注点转换到 $[-1,1]$ 区间：

$$
x_{norm} = \frac{x}{320} - 1
$$

$$
y_{norm} = 1 - \frac{y}{112}
$$

横坐标根据 640 像素宽度归一化。纵坐标根据 224 像素高度归一化，同时翻转方向，使图像越上方的点具有越大的归一化值。

对应实现如下：

```python
def get_x(value, width):
    return (value * 224.0 / 640.0 - width / 2) / (width / 2)

def get_y(value, height):
    return ((224 - value) - height / 2) / (height / 2)
```

## ResNet18 多任务模型

训练脚本加载 ImageNet 预训练的 ResNet18，并把分类头替换为三个输出：

```python
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 3)
```

每张图片经过颜色扰动、224×224 缩放、Tensor 转换和 ImageNet 标准化：

```python
image = self.color_jitter(image)
image = transforms.functional.resize(image, (224, 224))
image = transforms.functional.to_tensor(image)
image = transforms.functional.normalize(
    image,
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225],
)
```

水平翻转及对应的 $x$ 坐标取反已经在数据集类中实现，但当前训练入口传入的是 `random_hflips=False`。需要启用时，应先确认赛道标志、障碍物和比赛规则不会引入明显的左右不对称语义。

## 多任务损失函数

模型同时优化坐标回归和中线存在性分类：

$$
\mathcal{L} = \mathcal{L}_{coord} + \mathcal{L}_{conf}
$$

置信度分支使用 `BCEWithLogitsLoss`，直接接收第三个输出 logit：

$$
\mathcal{L}_{conf} = \operatorname{BCEWithLogits}(c, m)
$$

其中 $m\in\{0,1\}$ 表示当前图片是否存在有效中线。坐标损失只在正样本上计算：

$$
\mathcal{L}_{coord}
= \frac{1}{N_+}
  \sum_i m_i
  \left\lVert
  \hat{\mathbf{p}}_i-\mathbf{p}_i
  \right\rVert_2^2
$$

这样，无中线图片只训练置信度分支，不会使用占位坐标干扰回归头。

```python
pred_xy = outputs[:, :2]
pred_conf_logit = outputs[:, 2]

loss_conf = criterion_midline_conf(
    pred_conf_logit,
    target_has_midline.squeeze(1),
)

has_midline_mask = target_has_midline.squeeze(1).bool()
loss_coords = torch.tensor(0.0, device=device)

if has_midline_mask.any():
    loss_coords = criterion_coords(
        pred_xy[has_midline_mask],
        target_xy[has_midline_mask],
    )

combined_loss = loss_coords + loss_conf
```

当前配置使用 Adam、批大小 32，并训练 100 个 epoch。测试集总损失最低的权重保存为 `best_line_follower_model_xy_conf.pt`。

```bash
python utils/train.py
```

## 在 PC 上验证模型

部署前应先使用 `detect.py` 检查单张图片或视频，确认坐标归一化、ROI 和置信度逻辑一致。

```bash
python utils/detect.py \
  --weights best_line_follower_model_xy_conf.pt \
  --source test.mp4 \
  --output track_result.mp4 \
  --confidence_threshold 0.5 \
  --fps 30
```

验证脚本会把输入缩放到 640×480，截取底部 640×224 区域，再缩放为模型需要的 224×224。置信度高于阈值时，预测中心点会绘制到原始画面。

除了观察视频，还应统计以下指标：

- 有效中线样本的横向像素 MAE
- 中线存在性分类的准确率、召回率和 F1
- 不同光照与弯道类型下的误差分布
- 板端端到端延迟与稳定帧率

横向误差通常比纵向误差更直接影响转向控制，因此不应只观察总 MSE。

## 导出 ONNX 与准备 BPU 模型

`export.py` 使用固定输入尺寸导出 opset 11 ONNX：

```bash
python utils/export.py
```

模型接口为：

| 项目 | 值                                      |
| ---- | --------------------------------------- |
| 输入 | `1 × 3 × 224 × 224`                     |
| 输出 | `x_norm, y_norm, confidence_logit`      |
| ONNX | `best_line_follower_model_xy_conf.onnx` |

仓库通过 `.gitignore` 排除了 `.pt`、`.onnx` 和 `.bin` 文件，也没有包含 PTQ 编译配置。因此，ONNX 到 RDK X5 `.bin` 的量化与编译需要在项目外完成。转换时必须保证输入颜色空间、ImageNet 归一化和输出顺序与训练代码一致。

生成模型后，将其放到项目的 `config/` 目录：

```text
config/
└── race_track_detection.bin
```

## RDK X5 上的图像预处理

节点订阅 `/hbmem_img` 后，首先检查巡线是否启用。关闭状态下会直接返回，不再进行图像预处理和 BPU 推理。

摄像头输入是 640×480 NV12。代码把底部 640×224 区域裁剪并缩放到 224×224：

```cpp
hbDNNRoi roi;
roi.left = 0;
roi.top = 480 - 224;
roi.right = 640 - 1;
roi.bottom = 480 - 1;

cv::Mat img_mat(
    msg->height * 3 / 2,
    msg->width,
    CV_8UC1,
    static_cast<void *>(msg->data.data()));

cv::Range rows(roi.top, 480);
cv::Range cols(roi.left, 640);

cv::Mat crop = hobot_cv::hobotcv_crop(
    img_mat,
    msg->height,
    msg->width,
    224,
    224,
    rows,
    cols);
```

随后将裁剪结果转换为 `NV12PyramidInput`，交给 `dnn_node`。模型被固定分配到 RDK X5 的 BPU Core 1：

```cpp
dnn_node_para_ptr_->model_file = model_path_;
dnn_node_para_ptr_->model_task_type = model_task_type_;
dnn_node_para_ptr_->task_num = 1;
dnn_node_para_ptr_->bpu_core_ids.push_back(HB_BPU_CORE_1);
```

这里将 640×224 ROI 直接缩放为 224×224，横向比例发生变化。由于训练和 PC 验证采用相同变换，模型可以学习这种映射，但任何新的数据处理流程也必须保持一致。

## 输出解析与坐标还原

BPU 输出包含两个归一化坐标和一个置信度 logit。解析时先把坐标限制到 $[-1,1]$，再对置信度执行 Sigmoid：

```cpp
float x_norm = output_data[0];
float y_norm = output_data[1];
float confidence_logit = output_data[2];

x_norm = std::max(-1.0F, std::min(1.0F, x_norm));
y_norm = std::max(-1.0F, std::min(1.0F, y_norm));
float confidence = sigmoid(confidence_logit);
```

横坐标还原到 640 像素宽度：

$$
x = 320(x_{norm}+1)
$$

纵坐标先还原到 224 像素高的 ROI，再加上原图中的裁剪偏移 256：

$$
y_{roi} = 112(1-y_{norm})
$$

$$
y_{image} = y_{roi} + 256
$$

对应代码如下：

```cpp
result->x = (x_norm * 112 + 112) * 640.0 / 224.0;
result->y = 224 - (y_norm * 112 + 112);

float x = result->x;
float y = result->y + 256;
```

## 发布赛道中心点

结果通过 `ai_msgs::msg::PerceptionTargets` 发布。消息中的目标类型为 `track_center`，点类型为 `midline_point`：

```cpp
ai_msgs::msg::Target target;
target.type = "track_center";

ai_msgs::msg::Point center;
center.type = "midline_point";

geometry_msgs::msg::Point32 point;
point.x = x;
point.y = y;
point.z = 0.0F;

center.point.emplace_back(point);
center.confidence.emplace_back(confidence);
target.points.emplace_back(center);
message.targets.emplace_back(target);
```

下游控制节点可以读取 $x$ 与图像中心 $x_c=320$ 的横向误差：

$$
e_x = x - x_c
$$

再通过 PID、纯跟踪或其他横向控制方法生成转角。这个感知节点只发布中心点，不直接控制舵机，因此感知与控制可以独立调试。

## 运行时暂停与恢复巡线

`/sign4return` 用于切换节点状态：

```cpp
if (sign_value == 5) {
    enable_lane_following_ = false;
} else if (sign_value == 6) {
    enable_lane_following_ = true;
}
```

手动暂停：

```bash
ros2 topic pub --once /sign4return std_msgs/msg/Int32 "{data: 5}"
```

恢复巡线：

```bash
ros2 topic pub --once /sign4return std_msgs/msg/Int32 "{data: 6}"
```

暂停后，图像回调会在预处理前退出，因此不仅停止发布结果，也避免继续占用 BPU。

## 编译与启动

克隆仓库并准备模型：

```bash
git clone https://github.com/JasperXzy/racing_track_detection.git
cd racing_track_detection

mkdir -p config
cp /path/to/race_track_detection.bin config/
```

编译 ROS2 package：

```bash
colcon build --symlink-install
source install/setup.bash
```

先启动摄像头：

```bash
ros2 launch origincar_bringup camera.launch.py
```

然后启动赛道中心点检测节点：

```bash
ros2 launch racing_track_detection racing_track_detection.launch.py
```

检查输出：

```bash
ros2 topic echo /racing_track_center_detection
```

需要 Web 可视化时，可以使用：

```bash
ros2 launch racing_track_detection racing_track_detection_web.launch.py
```

## 总结

这个项目把赛道跟随拆成了一条轻量且清晰的感知链路：

- 数据端只标注一个中心点，并加入无中线负样本
- 模型端使用 ResNet18 同时回归坐标与置信度
- 部署端直接处理 NV12 ROI，并在 RDK X5 BPU 上推理
- 系统端通过 ROS2 发布标准感知消息，并支持动态暂停与恢复

相较于完整赛道分割，中心点回归更容易达到实时运行要求，也更适合摄像头视角和赛道结构固定的小车竞速场景。真正决定系统稳定性的，不只是网络结构，还包括统一的数据裁剪、坐标定义、量化预处理和下游控制策略。

代码仓库：[JasperXzy/racing_track_detection](https://github.com/JasperXzy/racing_track_detection)
