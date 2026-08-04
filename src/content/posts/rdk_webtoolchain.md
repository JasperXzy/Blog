---
title: "从 ONNX 到 HBM：RDK WebToolChain 的设计与实现"
pubDatetime: 2026-08-04T12:00:00+08:00
featured: false
draft: false
category: Projects
description: "面向 RDK S100/S600 的本地 ONNX 检查、PTQ 编译、结果分析与板端验证工作台"
---

将一个训练完成的 ONNX 模型部署到 RDK 开发板，通常不只是执行一次模型转换命令。开发者还需要确认输入 Shape 与数据布局、准备校准集、配置量化与编译参数、分析转换日志和产物，最后再把 HBM 模型复制到开发板上验证。任何一个环节配置不一致，都可能得到“成功生成但无法正确运行”的模型。

[RDK WebToolChain](https://github.com/JasperXzy/RDK_WebToolChain) 将这些分散的步骤组织成一个运行在本机的 Web 工作台。它面向 D-Robotics RDK S100 和 S600，通过浏览器完成 ONNX 检查、校准数据管理、PTQ 转换、结果分析、任务比较和 HBM 板端测试，底层使用 OpenExplorer 3.7.0。

RDK WebToolChain 并不是新的模型编译器，也不是云端模型转换服务。它解决的是模型部署流程中的工程问题：如何约束配置、隔离任务、保存上下文，并让每次转换都能够被解释和复现。

## 为什么需要一个模型转换工作台

OpenExplorer 已经提供完整的模型检查和量化编译能力，但在实际项目中，一条可用的部署链路还需要处理以下问题：

- S100 与 S600 使用不同的目标架构和编译约束，错误组合可能直到编译阶段才暴露。
- 模型输入、训练输入、Runtime 输入、颜色空间和归一化参数之间存在大量交叉约束。
- 校准图片需要按照与训练一致的方式预处理，直接使用 NPY 时还要检查 Shape、dtype 和数值有效性。
- PTQ 编译时间较长，浏览器刷新或服务重启不能导致任务状态和日志丢失。
- ONNX、YAML、HBM、量化报告和性能报告需要与本次配置建立确定关联。
- 板端验证涉及 SSH 凭据、模型上传和远端命令，不能简单地向 Web 页面开放 Shell。

因此，项目的重点不是把命令行包装成按钮，而是在 OpenExplorer 外部建立一层受控、可恢复、可追溯的工作流。

## 系统架构

RDK WebToolChain 采用本地单机部署。浏览器只与 Controller 通信，Controller 负责项目数据、任务状态和 Docker 调度。真正执行 OpenExplorer 的 Runner 是按任务创建的短生命周期容器。

```text
Browser
   │
   │ HTTP / SSE
   ▼
Controller
   ├── FastAPI Web/API
   ├── SQLite 元数据与持久任务队列
   ├── Assets / Runs / Cache Named Volume
   ├── Docker Engine Gateway
   │       │
   │       └── OpenExplorer Runner Container
   │               ├── ONNX Inspect
   │               ├── hb_compile Check / Compile
   │               ├── HBRuntime / hb_verifier
   │               └── Artifact Collection
   │
   └── SSH / SFTP Gateway
           └── RDK S100 / S600
                   └── hrt_model_exec
```

这种拆分将控制面与执行面分离：Controller 不直接加载工具链执行模型转换，Runner 也不依赖 Web、数据库或业务服务。两者通过版本化 JSON 请求、事件和结果文件通信，OpenExplorer 的版本差异则收敛在独立 Adapter 中。

项目默认只监听 `127.0.0.1`，模型、校准数据、日志和产物保存在本机 Docker Volume 中。只要 OpenExplorer 镜像已经存在，核心转换流程可以在不访问外网的情况下运行。

## 从 ONNX 到 HBM 的完整流程

一次标准转换由六个阶段组成。

### 1. 导入并检查 ONNX

用户首先创建项目并上传 ONNX。系统计算文件的 SHA-256，将模型保存到内容寻址的资产目录，然后创建独立的模型检查任务。

检查任务会解析模型的 IR Version、opset、输入输出、算子统计、静态或动态维度以及 External Data 依赖。只有文件哈希保持一致且检查状态为 `READY` 的模型，才能进入正式转换流程。

这样做可以避免由 Controller 直接解析不可信模型，也能够确保后续使用的模型与检查结果确实是同一份文件。

### 2. 准备校准数据

系统支持图片、单输入 NPY 和多输入 NPY 三类校准数据。

对于图片，用户通过声明式 Recipe 组合 Resize、Center Crop、Letterbox、颜色转换、HWC/CHW 转换、dtype 转换和归一化步骤。提交前可以查看处理结果、Shape、dtype、最小值、最大值和均值，确认它与模型输入一致。

对于 NPY，系统使用受限方式读取文件，并检查以下内容：

- dtype、字节序与内存布局是否合法。
- 维度和 Shape 是否符合模型输入。
- 是否包含 NaN、Inf、对象或结构化数据。
- Header、Payload 大小与文件哈希是否一致。

多输入模型使用 `<input_name>/<sample>.npy` 的目录结构。每个输入目录必须拥有相同的样本集合，输入名必须与 ONNX 完全一致，因此执行过程不依赖文件遍历顺序。

校准集定稿后，源文件清单、预处理 Recipe、生成结果和哈希共同构成不可变版本。更换图片或调整预处理都会生成新的校准版本，而不是覆盖历史数据。

### 3. 选择目标平台并生成配置

S100 与 S600 的平台差异由 Target Profile 管理，而不是交给用户自由填写。

| 配置           | S100       | S600                   |
| -------------- | ---------- | ---------------------- |
| `march`        | `nash-e`   | `nash-p`               |
| BPU Core       | 1          | 1 或 2                 |
| `max_l2m_size` | 固定为 0   | 关闭、自动或平台允许值 |
| Profile 快照   | 随任务冻结 | 随任务冻结             |

用户仍然可以配置模型输入、训练数据布局、Runtime 输入、Mean、Scale、编译优化和验证选项，但所有字段都要经过类型检查和跨字段校验。例如，NV12 输入不能使用奇数宽高，动态维度必须给出目标值，Mean 与 Scale 的数量必须和通道配置一致。

Controller 使用类型化 JSON 作为配置的事实来源，再由 OpenExplorer 3.7.0 Adapter 生成最终 YAML。提交前展示 YAML 预览，同时区分平台锁定字段、用户字段和默认字段。

### 4. 冻结任务并执行 PTQ 转换

提交任务时，系统不会只保存一份 YAML，而是冻结一组完整快照：

- ONNX ID、文件哈希和输入信息。
- 校准集版本、样本清单和清单哈希。
- 规范化配置与生成的 YAML。
- Target Profile ID、版本和能力快照。
- Runner 镜像引用及不可变 Image ID。
- 应用、Runner 合约和工具链版本。

修改任何配置都会创建新的 Run；使用同一快照重试则创建新的 Attempt。这样既不会覆盖失败记录，也能清楚地区分“更改配置后重新转换”和“相同配置下重新执行”。

默认执行链路为：

```text
QUEUED
  → PROVISIONING
  → INSPECTING
  → CHECKING
  → PREPROCESSING
  → COMPILING
  → VERIFYING
  → COLLECTING
  → SUCCEEDED / FAILED / CANCELLED / INTERRUPTED
```

SQLite 是任务状态的事实来源。Controller 先持久化状态，再执行 Docker 操作。任务日志与阶段事件通过 SSE 推送到页面，并支持按序号续接，因此刷新页面不会丢失已经产生的日志位置。

### 5. 验证并分析结果

成功生成 HBM 只是结果判定的一部分。系统还会核对 Runner 退出状态、`hb_compile` 退出码、HBM 文件大小、必要日志、结果文件和 Artifact Manifest。日志中出现某个成功字符串不会被单独视为任务成功。

启用数值验证后，Runner 会使用 HBRuntime 对浮点中间模型执行真实预处理输入，并通过 `hb_verifier` 检查不同阶段模型的数值一致性。结果页统一展示：

- HBM 文件、大小和 SHA-256。
- 输出节点量化余弦相似度。
- 静态 latency、FPS 和内存访问指标。
- 各阶段耗时、原始日志和工具链版本。
- YAML、量化报告、性能报告和结构化结果。

项目还可以比较同一模型版本的多次成功任务，直接查看平台、校准集、编译参数、HBM、性能和验证指标的差异。编译缓存以模型哈希、校准哈希、规范化配置、Profile、Adapter 和 Runner Image ID 共同生成 Key，避免不同上下文错误复用缓存。

### 6. 在 RDK 开发板上运行 HBM

静态编译指标不能替代开发板实测。RDK WebToolChain 将板端验证设计成独立任务，通过 SSH 和 SFTP 上传已经完成哈希复核的 HBM，再调用板端 `hrt_model_exec` 执行以下操作：

- `model_info`：读取模型输入输出和基本信息。
- `infer`：执行单图推理并保存输出 BIN。
- `perf`：采集 latency、FPS 和 Profile 结果。

转换 Runner 始终保持禁网，SSH 连接由 Controller 单独负责。首次连接开发板时，页面只展示观察到的 Host Key 指纹，用户核对并明确保存后才能继续。系统不会读取用户的 SSH Agent、默认私钥或 `known_hosts`，也不接受任意远端命令、工具路径和 Shell 参数。

设备平台还必须与 HBM 的 Target Profile 一致。例如，为 S600 编译的模型不能提交到登记为 S100 的设备上运行。

## Runner 的隔离边界

Controller 可以访问 Docker Socket，因此对它暴露的能力必须严格限制。普通任务请求不能指定镜像、EntryPoint、宿主机路径、网络模式、挂载、特权模式或任意命令。

每个 Runner 使用冻结的镜像 ID，并应用以下限制：

```text
network             none
root filesystem     read-only
Linux capabilities  drop ALL
security option     no-new-privileges
assets volume       read-only
runs/cache volume   read-write
temporary directory restricted tmpfs
CPU / memory / PID  limited
```

Controller 为容器生成固定名称和 Label，并在停止、恢复或清理前重新核对 Run ID 与 Attempt。即使 API 收到错误的容器 ID，也不会操作身份不匹配的容器。

这层边界不能消除 Docker Socket 本身的高权限属性，因此 RDK WebToolChain 仍然只适合本机单用户部署。但它可以阻止模型配置被转换成任意 Docker 参数，将工具能够执行的操作收敛到预定义的 OpenExplorer 工作流中。

## 一次真实的 S100/S600 转换基线

项目使用 ResNet18、20 张 ImageNet 校准图片和 OpenExplorer 3.7.0 对两种平台完成了隔离转换基线。下表中的性能数据来自编译器静态估算，不是开发板实测值。

| 指标               |         S100 |         S600 |
| ------------------ | -----------: | -----------: |
| `march`            |     `nash-e` |     `nash-p` |
| BPU Core           |            1 |            2 |
| 最终量化余弦相似度 |     0.994883 |     0.994565 |
| HBM 大小           | 12,203,840 B | 13,137,104 B |
| 静态延迟           |       411 μs |     278.4 μs |
| 静态 FPS           |      2433.29 |      3592.52 |

两次任务均生成 HBM、YAML、模型检查结果、校准 Manifest、量化信息、工具日志和静态性能报告，并重新核验 Artifact Manifest 中记录的文件大小与 SHA-256。

这组数据的意义不是比较两块开发板的真实性能，而是验证同一套 Controller、Runner 合约和产物归集逻辑能够在 S100 与 S600 两种 Profile 下完成端到端闭环。

## 安装与启动

项目目前要求 Linux `amd64` 主机、Docker Engine、Docker Compose v2，以及已经加载到本机的 OpenExplorer 3.7.0 CPU 基础镜像：

```text
ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0
```

首先构建 CPU Runner：

```bash
docker build \
  --file runner/Dockerfile.cpu \
  --tag rdk-webtoolchain/oe-runner-cpu:oe3.7.0-app0.1 \
  .
```

然后执行安装脚本：

```bash
./scripts/rdkwt.sh install
```

脚本会生成本地配置、检查 Docker 环境、设置 Docker Socket GID、构建 Controller 并启动服务。默认访问地址为：

```text
http://127.0.0.1:8080/
```

遇到环境问题时，可以先执行：

```bash
./scripts/rdkwt.sh doctor
```

项目还提供 `up`、`down`、`backup`、`restore`、`upgrade` 和 `diagnostics` 等维护命令。停止服务不会删除数据卷，备份包则通过成员清单、文件大小和 SHA-256 保证完整性。

## 当前边界

RDK WebToolChain 当前聚焦 ONNX、PTQ、HBM、OpenExplorer 3.7.0 和 RDK S100/S600，并不包含模型训练、数据标注、QAT 或通用模型发布服务。

CPU Runner 是已经完成真实转换门禁的规范路径。GPU 控制面已经预留，但在兼容 OpenExplorer 3.7.0 的 GPU、Driver 和 Container Toolkit 组合完成实机验证前，不应把它描述为已验证能力。

HBRuntime 与 `hb_verifier` 提供的是数值冒烟和阶段回归，也不能替代带标签业务数据集上的最终精度评估。板端任务链路已经实现，但具体设备上的最终结果仍应在目标 S100 或 S600 上完成验证。

## 总结

RDK WebToolChain 将模型部署中最容易散落的配置、数据、命令、日志和产物统一到一个本地工作流中。它保留了 OpenExplorer 的转换能力，同时通过 Target Profile、不可变快照、隔离 Runner、内容哈希和受限 SSH，把一次临时的模型转换变成可以恢复、比较和复现的工程过程。

对于需要反复调整输入、校准策略和编译参数的 RDK 项目，这种工作台的价值并不只是减少命令输入，而是让每一份 HBM 都能回答三个问题：它由什么生成、为什么得到这个结果，以及如何再次得到相同结果。
