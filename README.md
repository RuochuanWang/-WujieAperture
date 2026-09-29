# 无界光圈 WujieAperture

> 面向后期景深控制的计算摄影平台  
> **拍摄已结束，光圈仍可选择。**

无界光圈（WujieAperture）是一套面向单张图像后期景深控制的计算摄影系统。项目以单目深度估计、显式散焦建模与遮挡感知渲染为核心，使已经拍摄完成的清晰图像能够在后期指定焦平面，并以摄影语义调节等效光圈和景深效果。

本仓库包含网页工作台、FastAPI 后端、DepthPro 深度估计适配、BokehStream 相关渲染代码、训练与评测代码、示例图片以及模型权重。

## 主要能力

- **单图景深控制**：输入一张普通 RGB 图像即可完成后期景深渲染。
- **点击指定焦平面**：直接在图像上点击目标位置，系统根据局部深度确定目标焦平面。
- **连续等效光圈**：VATD 配置支持约 f/1.2–f/16 的连续控制；EBB 配置用于 f/1.8 专项渲染。
- **自定义处理分辨率**：工作台提供 512 / 640 / 768 / 1024 / 1280 / 1536 / 1920 / 2048 px 推荐档位，并支持 256–2048 px 自定义最长边。
- **DepthPro 深度估计**：自动生成单目深度结果，并在一次深度估计后复用于后续光圈调整。
- **本地运行**：前后端均可在本机或内网环境部署，图像无需上传到第三方推理 API。
- **结果检查与下载**：支持原图、深度图、渲染结果和滑动对比视图，并可下载 PNG。

## 技术路线

当前网页推理链路可以概括为：

```text
输入 RGB 图像
    │
    ▼
DepthPro 单目深度估计
    │
    ▼
逆深度归一化 / 焦平面确定
    │
    ▼
光圈参数 → CoC / 散焦半径
    │
    ▼
遮挡感知的分层显式渲染
    │
    ▼
散景结果 PNG
```

仓库中的 `method/` 还保留了研究阶段使用的显式渲染器、残差校正器、修复先验和损失函数等模块。为了保证网页演示链路的可解释性，当前 Web 工作台默认输出显式物理渲染结果。

## 项目结构

```text
.
├── competition_app/              # FastAPI 服务与前端网页
│   ├── main.py                   # API 与页面路由
│   ├── inference.py              # 推理管线
│   ├── depth_pro_backend.py      # DepthPro 适配
│   └── static/                   # HTML / CSS / JavaScript / 展示资源
├── method/                       # BokehStream / 渲染与研究模块
├── dataset/                      # 数据集读取与预处理
├── scripts/                      # 安装、启动、环境检查脚本
├── tests/                        # Web 与推理链路测试
├── checkpoints/
│   └── depth_pro.pt              # DepthPro 权重（Git LFS）
├── weights/
│   ├── VATD_weight.pth           # VATD 权重（Git LFS）
│   └── EBB_weight.pth            # EBB 权重（Git LFS）
├── train.py                      # 训练入口
├── train_vatd_physical_v3.yml    # VATD 配置
├── render_photos.py              # 图像渲染工具
├── requirements.txt              # 完整 Python 依赖
├── requirements-app.txt          # 仅 Web 应用依赖
├── requirements-dev.txt          # 测试依赖
├── Dockerfile
└── docker-compose.yml
```

## 环境要求

推荐环境：

- Windows 10/11 或 Linux
- Python 3.10
- NVIDIA GPU（推荐）
- CUDA 兼容的 PyTorch
- Git
- Git LFS

模型推理可在 CPU 上运行部分流程，但完整散景渲染更适合使用 CUDA GPU。

## 获取仓库

本仓库包含大模型权重，使用 Git LFS 管理。首次克隆前建议确认 Git LFS 已安装。

```bash
git lfs install
git clone https://github.com/RuochuanWang/-WujieAperture.git
cd -WujieAperture
git lfs pull
```

其中 `checkpoints/depth_pro.pt` 约 1.9 GB，请确保本地磁盘空间和 Git LFS 配额充足。

## 安装

### 方案一：自动安装脚本

Windows PowerShell：

```powershell
.\scripts\setup.ps1
```

Linux / macOS：

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

安装脚本会创建 `.venv`，安装应用依赖，并检查 DepthPro 与项目模型权重。

### 方案二：手动安装

建议先创建 Python 3.10 虚拟环境。

```bash
python -m venv .venv
```

Windows：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux / macOS：

```bash
source .venv/bin/activate
```

NVIDIA CUDA 12.8 环境可先安装对应 PyTorch：

```bash
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
```

再安装其余依赖：

```bash
pip install -r requirements.txt
```

如果 `depth_pro` 未成功安装，可单独执行：

```bash
pip install "depth_pro @ git+https://github.com/apple/ml-depth-pro.git@9efe5c1"
```

## 启动网页

先检查环境：

```bash
python scripts/verify_install.py
```

启动服务：

```bash
python -m competition_app
```

默认访问：

```text
http://127.0.0.1:7860
```

Windows 文件包中也提供了 `启动无界光圈网页.bat`。如果其中 Python 路径与本机环境不同，请按实际安装位置修改。

## 工作台使用流程

1. 上传 JPG、PNG、WebP、BMP 或 TIFF 图像。
2. 选择处理尺寸；显存紧张时建议 512–768 px，常规演示建议 768–1024 px。
3. 选择 VATD 或 EBB 渲染配置。
4. 调整目标光圈。
5. 点击图像中的目标主体或目标区域以指定焦平面。
6. 等待 DepthPro 深度估计与散景渲染完成。
7. 在“原图 / 深度 / 渲染结果 / 对比”之间切换检查。
8. 下载深度图或最终 PNG。

## 常用环境变量

可参考 `.env.example`：

```text
BOKEH_DEVICE=auto
BOKEH_MAX_LONG_SIDE=768
BOKEH_MAX_CUSTOM_LONG_SIDE=2048
BOKEH_AMP=true
BOKEH_MAX_UPLOAD_MB=20

BOKEH_VATD_CHECKPOINT=weights/VATD_weight.pth
BOKEH_EBB_CHECKPOINT=weights/EBB_weight.pth
DEPTH_PRO_CHECKPOINT=checkpoints/depth_pro.pt

BOKEH_HOST=127.0.0.1
BOKEH_PORT=7860
```

其中 `BOKEH_MAX_CUSTOM_LONG_SIDE` 用于限制工作台允许的最大手动处理尺寸。

## 测试

安装开发依赖：

```bash
pip install -r requirements-dev.txt
```

运行：

```bash
pytest -q
```

## 模型文件

仓库使用 Git LFS 管理以下权重：

```text
checkpoints/*.pt
weights/*.pth
```

当前主要文件：

- `checkpoints/depth_pro.pt`：Apple DepthPro 深度估计权重。
- `weights/VATD_weight.pth`：VATD 连续光圈配置权重。
- `weights/EBB_weight.pth`：EBB f/1.8 专项配置权重。

DepthPro 的原始项目与模型版权归其原作者所有；使用时请同时遵守其对应许可与使用条款。

## 当前限制

单目景深估计与后期散景渲染仍可能在以下场景出现误差：

- 透明、半透明物体；
- 镜面和强反射区域；
- 发丝、树枝、栏杆等极细结构；
- 多层遮挡与复杂前后景交界；
- 极端高光散景；
- 深度估计本身存在明显错误的区域。

此外，当前系统主要根据图像内容和模型标定进行等效光圈控制，并不完整复现任意真实镜头的传感器尺寸、焦距、叶片形状和像差特性。

## 研究与应用定位

无界光圈面向后期景深控制、计算摄影实验、影像创作、教学演示和本地化部署。项目重点不是通用生成式重绘，而是在尽量保留原图内容的前提下，通过显式空间与光学参数控制改变景深表达。

---

**WujieAperture / 无界光圈**  
面向后期景深控制的计算摄影平台。