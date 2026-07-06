# tianjiaying 环境 GPU 诊断与修复记录

日期：2026-07-05
状态：已修复

## 1. 当前结论

`tianjiaying` 现在可以使用 GPU。

这次问题不是服务器没有 GPU，也不是 OUGP 代码问题，而是 `tianjiaying` conda 环境里的 PyTorch CUDA 依赖不完整，并且 Python 会读取 `~/.local/lib/python3.10/site-packages`，导致环境隔离不干净。

已完成修复：

- 在 `tianjiaying` 环境内部重装 `torch==2.7.1+cu118`、`torchvision==0.22.1+cu118`、`torchaudio==2.7.1+cu118`。
- 让 CUDA 11.8 相关依赖完整安装到 `tianjiaying` 环境内部。
- 设置 `tianjiaying` 激活后默认 `PYTHONNOUSERSITE=1`，避免继续优先读取 `~/.local` 用户级 Python 包。
- 用 1 张 RTX 3090 做了 CUDA 张量计算验证。

## 2. 诊断证据

服务器 GPU 正常，非沙箱环境可见 8 张 RTX 3090：

```text
0, NVIDIA GeForce RTX 3090, 535.309.01, 20371 MiB, 24576 MiB
1, NVIDIA GeForce RTX 3090, 535.309.01, 20371 MiB, 24576 MiB
2, NVIDIA GeForce RTX 3090, 535.309.01, 20507 MiB, 24576 MiB
3, NVIDIA GeForce RTX 3090, 535.309.01, 12 MiB, 24576 MiB
4, NVIDIA GeForce RTX 3090, 535.309.01, 12 MiB, 24576 MiB
5, NVIDIA GeForce RTX 3090, 535.309.01, 12 MiB, 24576 MiB
6, NVIDIA GeForce RTX 3090, 535.309.01, 12 MiB, 24576 MiB
7, NVIDIA GeForce RTX 3090, 535.309.01, 12 MiB, 24576 MiB
```

修复前，`tianjiaying` 里 `import torch` 失败，关键报错包括：

```text
libcudart.so.11.0: cannot open shared object file
ValueError: libnvJitLink.so.*[0-9] not found in the system path
```

禁用用户级包后，报错进一步指向环境内部缺依赖：

```text
ValueError: libcublas.so.*[0-9] not found in the system path
```

说明当时不是 GPU 硬件问题，而是 `tianjiaying` 环境内部 CUDA runtime 依赖不完整。

## 3. 已执行的修复

重装 PyTorch CUDA 11.8 wheel：

```bash
PYTHONNOUSERSITE=1 /home/shizitong/miniconda3/envs/tianjiaying/bin/python -m pip install \
  --no-user \
  --force-reinstall \
  --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.7.1+cu118 \
  torchvision==0.22.1+cu118 \
  torchaudio==2.7.1+cu118
```

设置环境激活时默认关闭 user site：

```bash
/home/shizitong/miniconda3/bin/conda env config vars set -n tianjiaying PYTHONNOUSERSITE=1
```

以后如果你已经打开了一个旧 shell，需要重新激活环境：

```bash
conda deactivate
conda activate tianjiaying
```

## 4. 修复后验证结果

PyTorch 和 CUDA 依赖现在位于 `tianjiaying` 环境内部：

```text
torch 2.7.1+cu118
Location: /home/shizitong/miniconda3/envs/tianjiaying/lib/python3.10/site-packages

nvidia-cublas-cu11 11.11.3.6
Location: /home/shizitong/miniconda3/envs/tianjiaying/lib/python3.10/site-packages

nvidia-cuda-runtime-cu11 11.8.89
Location: /home/shizitong/miniconda3/envs/tianjiaying/lib/python3.10/site-packages
```

依赖检查通过：

```text
No broken requirements found.
```

用 1 张 GPU 验证：

```bash
CUDA_VISIBLE_DEVICES=3 /home/shizitong/miniconda3/bin/conda run -n tianjiaying python -c \
  "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"
```

输出：

```text
torch= 2.7.1+cu118
cuda_build= 11.8
cuda_available= True
device_count= 1
device0= NVIDIA GeForce RTX 3090
```

还做了一个最小 CUDA 矩阵乘法，能够正常完成：

```text
matmul_mean= 0.020861856639385223
```

## 5. 后续运行 OUGP 的建议

进入项目：

```bash
cd /home/shizitong/tianjiaying/research/ougp
conda activate tianjiaying
```

单卡运行时建议显式指定一张空闲卡，例如：

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src python scripts/run_case_study.py \
  --dataset cora \
  --epochs 80 \
  --warmup-epochs 10 \
  --variants dense graph_only param_only dual_static ougp_no_cross ougp \
  --seeds 0 \
  --out-dir experiments/gpu_test_cora_seed0 \
  --graph-gamma 2.0 \
  --param-gamma 2.0 \
  --write-beta 0.25 \
  --device cuda
```

最多使用 4 张卡时：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ...
```

当前建议优先用 1 张卡跑 OUGP case study，因为 Cora 很小，多卡不会明显加速，反而会增加环境和并行复杂度。

## 6. 给导师或维护同学看的简短说明

```text
tianjiaying conda 环境的 GPU 问题已修复。

原问题：torch 2.7.1+cu118 装在 conda env 中，但 CUDA 11.8 依赖不完整，并且 Python 默认读取 ~/.local/lib/python3.10/site-packages，导致环境隔离被破坏。import torch 时出现 libcudart / libnvJitLink / libcublas 找不到的问题。

修复：在 tianjiaying env 内用 PyTorch cu118 index 强制重装 torch/torchvision/torchaudio，并安装完整 nvidia-cu11 依赖；同时设置 conda env config vars：PYTHONNOUSERSITE=1。

验证：CUDA_VISIBLE_DEVICES=3 conda run -n tianjiaying python 中 torch.cuda.is_available() == True，device_count == 1，设备为 RTX 3090，CUDA 张量计算正常。
```
