# 实验八：差分隐私生成模型

本项目完成课程实验八的必做任务，并实现了面向完整 MNIST 0–9 的 DP-WGAN 改进。仓库只保留两组正式结果：

- **必做内容**：DP-SGD + DCGAN，生成数字 8。
- **改进内容**：DP-ACWGAN-CP，条件生成数字 0–9，并解决类别模式坍缩。

## 目录结构

```text
differential-privacy-dcgan/
├── README.md
├── requirements.txt
├── scripts/
│   └── run_required.ps1
├── src/
│   ├── required/
│   │   └── train_dp_dcgan.py
│   └── improved/
│       ├── dp_wgan_models.py
│       ├── train_dp_wgan.py
│       ├── refine_dp_wgan.py
│       ├── train_mnist_evaluator.py
│       └── evaluate_dp_wgan.py
└── results/
    ├── required/
    │   ├── model/generator_last.pt
    │   ├── sample/generated.png
    │   ├── metrics/training_metrics.csv
    │   └── summary.json
    └── improved/
        ├── model/dp_wgan_refined.pt
        ├── sample/generated.png
        ├── evaluators/
        └── metrics/
```

## 环境

本次实际运行环境：

- Python 3.10
- PyTorch 2.4.1 + CUDA 12.4
- torchvision 0.19.1
- Opacus 1.5.4
- NVIDIA GeForce RTX 4060 Laptop GPU

安装依赖：

```powershell
python -m pip install -r .\requirements.txt
```

## 一、必做内容：DP-SGD + DCGAN

### 原理

判别器读取 MNIST 中的真实数字 8。Opacus 对每条样本的判别器梯度进行裁剪并加入高斯噪声；生成器只通过私有判别器获得训练信号。

正式配置：

| 项目 | 值 |
|---|---:|
| 数据 | MNIST 数字 8，共 5,851 张 |
| Epoch / Batch | 10 / 64 |
| 噪声乘数 | 1.0 |
| 最大逐样本梯度范数 | 1.0 |
| epsilon / delta | 1.93239 / 1e-5 |
| 训练时间 | 45.59 秒 |

### 运行

```powershell
.\scripts\run_required.ps1
```

或直接执行：

```powershell
.\.venv\Scripts\python.exe .\src\required\train_dp_dcgan.py `
  --data-root .\data `
  --output-dir .\runs\required `
  --epochs 10 --batch-size 64 --target-digit 8 `
  --noise-multiplier 1.0 --max-grad-norm 1.0 `
  --delta 1e-5 --device cuda
```

### 必做结果

- 模型：`results/required/model/generator_last.pt`
- 最终样本：`results/required/sample/generated.png`
- 训练指标：`results/required/metrics/training_metrics.csv`
- 配置摘要：`results/required/summary.json`

## 二、改进内容：DP-ACWGAN-CP

### 改进原因

基础方法容易出现类别模式坍缩，因此改进模型引入条件控制、辅助分类与 EMA，以提高类别覆盖和生成稳定性。

主要改进：

1. 使用覆盖完整图像的 Projection Critic。
2. 增加辅助分类头，直接约束真实图和生成图的类别。
3. 将 Wasserstein、Critic 分类和 Generator 分类损失权重设为 `0.01 / 1.0 / 2.0`。
4. WGAN 参数裁剪仅作用于分数分支，避免压制分类头。
5. 修复 Opacus 多次前向时未使用分类输出导致的逐样本梯度错误。
6. 使用 EMA Generator。
7. 使用不读取私有训练样本的公开分类器后处理，并通过不同结构的 ConvNet 交叉验证。

损失函数：

```text
L_C = 0.01 * (mean(C(fake,y)) - mean(C(real,y)))
      + CE(class(real), y)

L_G = -0.01 * mean(C(fake,y))
      + 2.0 * CE(class(fake), y)
```

### 正式训练

```powershell
.\.venv\Scripts\python.exe .\src\improved\train_dp_wgan.py `
  --data-root .\data `
  --output-dir .\runs\improved `
  --epochs 40 --batch-size 128 --latent-size 128 `
  --generator-features 64 --critic-features 32 `
  --critic-steps 3 `
  --generator-lr 1e-4 --critic-lr 1e-4 `
  --wasserstein-weight 0.01 `
  --critic-class-weight 1.0 --generator-class-weight 2.0 `
  --weight-clip 0.02 --max-grad-norm 1.0 `
  --target-epsilon 8 --delta 1e-5 `
  --ema-decay 0.99 --save-every 5 --device cuda
```

正式 DP 训练耗时 1112.25 秒，最终隐私预算为：

```text
epsilon = 7.994994171548938
delta   = 1e-5
sigma   = 0.576934814453125
```
### 评估

```powershell
.\.venv\Scripts\python.exe .\src\improved\evaluate_dp_wgan.py `
  --checkpoint .\results\improved\model\dp_wgan_refined.pt `
  --evaluator-checkpoint .\results\improved\evaluators\lenet_mnist_best.pt `
  --data-root .\data `
  --output-dir .\runs\evaluation `
  --num-samples 10000 --num-real 10000 `
  --batch-size 256 --device cuda
```

### 改进结果

所有指标使用 10,000 张生成样本、10,000 张真实测试样本和随机种子 2026。

| 模型 / 评估器 | 条件一致性 | 稳健覆盖 | 归一化熵 | 特征 FID |
|---|---:|---:|---:|---:|
| 最终改进 / LeNet | **100.00%** | **10/10** | **1.0000** | **464.98** |
| 最终改进 / 独立 ConvNet | **100.00%** | **10/10** | **1.0000** | **513.57** |

LeNet 真实测试准确率为 99.19%，独立 ConvNet 为 98.72%。不同评估器使用的特征空间不同，因此两种 FID 不能直接横向比较。

改进结果文件：

- 最终模型：`results/improved/model/dp_wgan_refined.pt`
- 最终样本：`results/improved/sample/generated.png`
- LeNet 指标：`results/improved/metrics/evaluation_lenet.json`
- ConvNet 交叉指标：`results/improved/metrics/evaluation_convnet.json`
- 类别分布：`results/improved/metrics/class_histogram_lenet.csv`、`class_histogram_convnet.csv`
- 评估器：`results/improved/evaluators/lenet_mnist_best.pt`、`convnet_mnist_best.pt`

## 指标说明

| 指标 | 含义 |
|---|---|
| `class_consistency_accuracy` | 评估器预测与请求标签一致的比例 |
| `robust_class_coverage_at_1_percent` | 生成占比至少 1% 的类别数量 |
| `normalized_class_distribution_entropy` | 类别分布均衡程度，越接近 1 越好 |
| `feature_fid` | 同一评估器特征空间中的生成分布距离，越低越好 |
| `epsilon` | 给定 delta 下的累计隐私损失 |
