# PES 最小可运行实验框架

本仓库提供一个二维势能面（PES）机器学习建模的最小可运行框架，包含：

- 解析势能与受力标签生成
- 训练集采样（uniform / grid / lhs / grad_importance）
- 两种力输出方式（direct / autograd）
- MLP 训练、验证、测试
- 基础可视化与结果导出（loss 曲线、采样分布、metrics）

## 环境

建议 Python 3.10+，并安装：

```bash
pip install torch numpy matplotlib
```

## 快速运行

在仓库根目录执行：

```bash
python pes_framework.py --quick --compare_sampling --output_dir outputs
```

运行后会在 `outputs/` 生成：

- `metrics.json`：各采样方法测试集能量/力 MAE
- `sampling_points.png`：采样点分布图
- `loss_*.png`：训练/验证 loss 曲线
- `model_*.pth`：模型参数

## 常用参数

- `--sampling_method {uniform,grid,lhs,grad_importance}`
- `--compare_sampling`：一次性比较四种采样方法
- `--force_mode {direct,autograd}`
- `--alpha` 与 `--lambda_force`：能量与力损失权重
- `--hidden_dims`：如 `64,64,64`
- `--activation {relu,tanh,gelu,sigmoid}`
- `--optimizer {sgd,adam,adamw,rmsprop}`
