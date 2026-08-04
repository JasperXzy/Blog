---
title: "在 RDK S100 上部署 YOLO26：C++ Runtime 的设计与实现"
pubDatetime: 2026-01-29T20:00:00+08:00
featured: false
draft: false
category: Projects
description: "RDK S100 的 YOLO26 Detect C++ Runtime，打通 NV12 预处理、UCP 推理、无锚框解码、NMS 与结果可视化"
---

本文记录如何在 RDK S100 上实现一套完整的 YOLO26 Detect C++ Runtime。项目从 HBM 模型出发，依次完成图像预处理、BPU 任务调度、检测头解码、NMS、坐标还原和结果可视化。模型导出、HBM 编译、Python 验证和 C++ 推理共同组成完整的部署链路。相较于 Python 示例，C++ Runtime 更适合继续集成到机器人、视频分析和低延迟边缘视觉应用中。

## 项目结构

C++ Runtime 由 3 个文件组成：

```text
samples/Vision/yolo26/runtime/cpp/
├── common/
│   └── common.h
└── detect/
    ├── CMakeLists.txt
    └── main.cc
```

三个文件的职责分别是：

| 文件             | 作用                                                 |
| ---------------- | ---------------------------------------------------- |
| `common.h`       | Letterbox、BGR 转 NV12、类别加载、检测结果打印与绘制 |
| `main.cc`        | HBM 加载、UCP 内存管理、BPU 推理、YOLO26 解码与 NMS  |
| `CMakeLists.txt` | 配置 OpenCV、`dnn`、`hbucp` 及 RDK S100 系统库       |

当前 C++ Runtime 聚焦 YOLO26 Detect 和 COCO 80 类单图推理。实例分割、姿态估计、分类与旋转框检测仍由已有的 Python Runtime 提供。

## 端到端推理链路

完整数据流如下：

```text
BGR 输入图像
    │
    ▼
Letterbox / Resize
    │
    ▼
BGR → YUV I420 → NV12
    │
    ├── Y Plane  : H × W
    └── UV Plane : H/2 × W/2 × 2
    │
    ▼
hbDNNInferV2
    │
    ▼
hbUCPSubmitTask → BPU → hbUCPWaitTaskDone
    │
    ▼
3 组 Box 张量 + 3 组 Class 张量
    │
    ▼
Logit 预筛选 → Anchor-Free 解码 → 分类别 NMS
    │
    ▼
逆 Letterbox 坐标映射
    │
    ▼
检测报告 + 可视化结果图
```

这条链路将模型相关逻辑与通用图像处理拆开。检测入口负责控制推理生命周期，`common.h` 则提供可以继续复用到其他 C++ 视觉示例中的基础函数。

## 为 BPU 导出 YOLO26 检测头

原始 Ultralytics 模型通常会在检测头内部完成张量拼接和部分后处理。为了让输出更适合 BPU 编译与板端解码，仓库中的导出脚本会替换 `Detect.forward`，直接导出三个尺度的原始 Box 与 Class 特征图。

其逻辑可以概括为：优先选择 End-to-End 的 one-to-one 分支；如果模型没有该分支，则回退到普通 Box 与 Class 分支。每个尺度分别执行预测，把 NCHW 转为 NHWC，然后按照 `[Box, Class]` 的顺序加入输出列表。

导出的 6 个张量全部使用 NHWC 布局：

| Stride | Box 输出  | Class 输出 |
| ------ | --------- | ---------- |
| 8      | `80×80×4` | `80×80×80` |
| 16     | `40×40×4` | `40×40×80` |
| 32     | `20×20×4` | `20×20×80` |

其中 Box 的 4 个通道表示网格中心到边界框四条边的距离 $[l,t,r,b]$，Class 的 80 个通道对应 COCO 类别 logits。

导出命令为：

```bash
python3 samples/Vision/yolo26/conversion/onnx_export/export_yolo26_detect_bpu.py \
  --weights yolo26n.pt \
  --output yolo26n_detect.onnx \
  --imgsz 640
```

脚本使用固定输入尺寸、ONNX opset 19，并关闭动态尺寸和图简化，从而保留面向 BPU 的检测头输出结构。

## 从 ONNX 编译为 HBM

RDK S100 使用 Nash-E 架构和 `.hbm` 模型格式。仓库提供的 `mapper.py` 会完成校准图片处理、编译配置生成以及 `hb_compile` 调用。

```bash
cd samples/Vision/yolo26/conversion

python3 mapper.py \
  --onnx yolo26n_detect.onnx \
  --cal-images ./cal_images \
  --march nash-e
```

转换配置的核心参数如下：

| 参数                   | 配置      |
| ---------------------- | --------- |
| 训练输入               | RGB、NCHW |
| 运行时输入             | NV12      |
| 像素缩放               | $1/255$   |
| 输入与输出额外 Padding | 关闭      |
| 编译目标               | Latency   |

校准阶段使用 RGB NCHW 浮点输入，并通过 $1/255$ 将像素缩放到 $[0,1]$。运行阶段则由编译器生成两个 NV12 输入平面，使 C++ Runtime 可以直接向 Y 与 UV 张量写入数据。

默认输出文件名包含模型、架构、输入尺寸和颜色格式，例如：

```text
yolo26n_detect_nashe_640x640_nv12.hbm
```

## CMake 与板端依赖

C++ 示例直接在 RDK S100 上构建。CMake 使用 C++11，并链接以下组件：

- OpenCV
- `dnn`
- `hbucp`
- `pthread`
- `rt`、`dl` 与 `m`

获取项目代码后进入检测目录编译：

```bash
git clone -b feat/yolo26-detect-cpp-runtime \
  https://github.com/JasperXzy/rdk_model_zoo_s.git
cd rdk_model_zoo_s

cd samples/Vision/yolo26/runtime/cpp/detect
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j4
```

## Letterbox 与坐标参数

C++ Runtime 默认使用 Letterbox，在保持原图宽高比的同时填充到模型输入尺寸。设原图尺寸为 $W_0×H_0$，模型输入为 $W×H$，缩放比例为：

$$
s=\min\left(\frac{W}{W_0},\frac{H}{H_0}\right)
$$

缩放后的尺寸为：

$$
W_r=\lfloor W_0s\rfloor,\qquad H_r=\lfloor H_0s\rfloor
$$

左右和上下填充偏移分别为：

$$
\Delta x=\frac{W-W_r}{2},\qquad
\Delta y=\frac{H-H_r}{2}
$$

填充区域使用 BGR 灰度值 127。这些缩放与偏移参数会一直保留到检测结果绘制阶段，用于把模型坐标还原到原图。

## 从 BGR 生成双平面 NV12 输入

OpenCV 先将 BGR 转换为 YUV I420。I420 的 U、V 平面连续存储，而 NV12 要求 UV 交错，因此需要在复制 Y 平面后手动交织色度数据。

内存重排可以概括为：完整复制 $H×W$ 的 Y 数据，再从 I420 中找到 U、V 平面的起始位置，按 `U0,V0,U1,V1,…` 的顺序写入剩余的 $H×W/2$ 字节。

转换完成后，C++ Runtime 为两个输入张量分别申请 UCP Cached Memory：

| 输入 | Shape         | 字节数  |
| ---- | ------------- | ------- |
| Y    | `1×H×W×1`     | $H×W$   |
| UV   | `1×H/2×W/2×2` | $H×W/2$ |

每次 `memcpy` 后执行 `HB_SYS_MEM_CACHE_CLEAN`，确保 BPU 读取到 CPU 刚写入的图像数据。

对应的内存操作顺序是 `hbUCPMallocCached → memcpy → HB_SYS_MEM_CACHE_CLEAN`。

## 加载 HBM 并提交 UCP 任务

模型加载沿用 D-Robotics C API：先用 `hbDNNInitializeFromFiles` 得到 Packed Handle，再读取模型名称列表，最后通过首个模型名称取得 DNN Handle。

输出张量按照 `alignedByteSize` 分配内存，随后通过 `hbDNNInferV2` 创建任务。调度器使用 `HB_UCP_BPU_CORE_ANY`，由运行时选择可用 BPU Core。

执行顺序为 `hbDNNInferV2 → hbUCPSubmitTask → hbUCPWaitTaskDone`。调度参数的 Backend 设置为 `HB_UCP_BPU_CORE_ANY`。

任务完成后，CPU 读取输出前执行 `HB_SYS_MEM_CACHE_INVALIDATE`，避免继续读取缓存中的旧数据。

## 根据 Shape 识别 6 个输出

模型导出顺序理论上是：

```text
Box_8, Cls_8, Box_16, Cls_16, Box_32, Cls_32
```

C++ Runtime 没有依赖具体的输出名称，而是根据空间尺寸和通道数重新建立顺序。空间尺寸决定 Stride，通道数为 4 时判定为 Box，否则判定为 Class。

实现上先用 $H/8$、$H/16$、$H/32$ 判断输出所属尺度，再根据通道数将张量写入 Box 或 Class 槽位。

这种方式让推理代码不依赖编译后张量名称，只要求模型继续保持三个尺度、4 通道 Box 和 80 通道 Class 的约定。

## 在 Logit 空间提前筛选候选框

如果对三个尺度的全部类别分数执行 Sigmoid，会产生大量没有必要的指数运算。由于 Sigmoid 单调递增，可以先把置信度阈值 $p_{min}$ 转换为 logit 阈值：

$$
z_{min}=-\ln\left(\frac{1}{p_{min}}-1\right)
$$

当 $p_{min}=0.25$ 时，$z_{min}\approx-1.0986$。代码先寻找每个网格的最大类别 logit，低于阈值时直接跳过，只对保留下来的候选计算 Sigmoid。

每个网格只需要遍历一次 80 个类别，记录最大 logit 与类别编号。最大值通过预筛选后，再计算这一个候选的 Sigmoid 分数。

## YOLO26 无锚框解码

每个网格输出到四条边的距离 $[l,t,r,b]$。对于位置 $(i,j)$ 和步长 $s$，网格中心定义为：

$$
c_x=j+0.5,\qquad c_y=i+0.5
$$

边界框坐标为：

$$
x_1=(c_x-l)s,\qquad y_1=(c_y-t)s
$$

$$
x_2=(c_x+r)s,\qquad y_2=(c_y+b)s
$$

三个尺度的候选框会按类别分别保存，再使用 OpenCV `NMSBoxes` 完成 NMS。默认置信度阈值为 0.25，IoU 阈值为 0.45。

## 还原到原图坐标

NMS 后的检测框仍位于模型输入坐标系。对于 Letterbox，原图坐标需要先减去填充偏移，再除以缩放比例：

$$
x'=\frac{x-\Delta x}{s},\qquad
y'=\frac{y-\Delta y}{s}
$$

绘制前再把坐标限制到图像边界，并根据类别选择颜色。程序同时在终端打印类别、置信度与 `xyxy` 坐标。

## 运行 C++ Runtime

可执行文件依次接受模型、输入图片、结果图片和类别文件路径：

```bash
./yolo26_detect \
  /absolute/path/yolo26n_detect_nashe_640x640_nv12.hbm \
  /absolute/path/bus.jpg \
  yolo26_detect_result.jpg \
  /absolute/path/coco_classes.names
```

四个参数均可省略。默认模型名为 `yolo26n_detect.hbm`，默认输入为仓库中的 `resource/assets/bus.jpg`，默认 COCO 标签来自 `datasets/coco/coco_classes.names`。

运行时会分别统计以下阶段：

- HBM 模型加载
- Letterbox 与 NV12 预处理
- BPU Forward
- 解码与 NMS 后处理

最后生成检测报告并将可视化结果保存为 `yolo26_detect_result.jpg`。示例不预设固定性能结论，具体延迟应以实际 S100 系统、模型版本和运行频率下的输出为准。

## 从 Python 验证到 C++ 集成

Python Runtime 更适合快速验证模型转换结果，C++ Runtime 则提供了更明确的内存生命周期和系统集成边界。两套实现保持相同的核心约定：

- 双平面 NV12 输入
- Stride 8、16、32 三尺度输出
- Box 与 Class 解耦检测头
- Logit 空间候选预筛选
- Anchor-Free LTRB 解码
- 分类别 NMS 与逆 Letterbox

因此可以先使用 Python 确认 HBM 模型输出正确，再切换到 C++ 检查板端内存、调度和后处理。这样能够把模型转换问题与 C++ 集成问题分开定位。

## 总结

这套项目为 YOLO26 Detect 打通了从 HBM 模型到 C++ 结果图的板端闭环：

- 导出端保留 6 个 BPU 友好的 NHWC 原始输出
- 编译端生成 Nash-E 的双平面 NV12 HBM 模型
- Runtime 使用 UCP Cached Memory 管理输入输出
- 推理端通过 `hbDNNInferV2` 和 UCP 调度 BPU 任务
- 后处理完成 Shape 匹配、Logit 筛选、无锚框解码和 NMS
- 可视化端还原 Letterbox 坐标并输出检测报告

这套实现让 YOLO26 不再只停留在 Python 验证脚本，而是具备了可以继续接入相机、ROS2、视频流和业务系统的 C++ 推理基础。

项目代码：[JasperXzy/rdk_model_zoo_s](https://github.com/JasperXzy/rdk_model_zoo_s/tree/feat/yolo26-detect-cpp-runtime)
