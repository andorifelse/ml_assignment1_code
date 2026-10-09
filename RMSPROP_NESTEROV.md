# RMSProp + Nesterov Momentum 更新策略说明

本文说明项目中新增的 `RMSPropNesterov` 优化方法，以及它与原有 `Momentum` 方法的区别。

## 1. 修改目标

项目原本已经支持 `normal`、`AdaGrad`、`Momentum`、`RMSProp` 和 `Adam`，但没有实现 RMSProp 与 Nesterov momentum 的组合。

新增方法严格按照以下步骤更新每一层参数：

1. 使用当前速度进行前瞻：

   \[
   \tilde{\theta}=\theta+\alpha v
   \]

2. 在前瞻参数 \(\tilde{\theta}\) 处执行前向传播和反向传播，得到梯度 \(g\)。
3. 更新 RMSProp 的平方梯度滑动平均：

   \[
   r=\rho r+(1-\rho)g\odot g
   \]

4. 更新速度：

   \[
   v=\alpha v-\frac{\epsilon}{\sqrt{r}+\delta}\odot g
   \]

5. 更新真实参数：

   \[
   \theta=\theta+v
   \]

其中除法和平方根均按元素计算，\(\delta\) 是防止除零的稳定常数。

## 2. 文件修改说明

### 2.1 `NN.py`

新增了三个可配置参数：

```python
rho=0.9
alpha=0.9
stability_constant=1e-8
```

含义如下：

| 参数 | 含义 | 常用范围 |
|---|---|---|
| `rho` | RMSProp 平方梯度的衰减率 | 0.9 到 0.99 |
| `alpha` | Nesterov 动量系数 | 0.8 到 0.99 |
| `stability_constant` | 防止除零的常数 | `1e-8` 或 `1e-7` |

当 `optimization_method='RMSPropNesterov'` 时，网络会初始化：

- `vW`、`vb`：权重和偏置的速度；
- `rW`、`rb`：权重和偏置的平方梯度累计量；
- 使用 Batch Normalization 时，还会初始化 `vGamma`、`vBeta`、`rGamma`、`rBeta`。

### 2.2 `nn_train.py`

对于 `RMSPropNesterov`，训练时不会直接在当前参数上计算梯度，而是：

```python
parameter = original_parameter + alpha * velocity
```

然后调用现有的 `nn_forward` 和 `nn_backward` 计算前瞻点梯度。梯度计算完成后，程序恢复原始参数，再交给 `nn_applygradient` 执行正式更新。

这样可以保证梯度确实是在 \(\theta+\alpha v\) 处计算，而不是误用当前参数 \(\theta\) 的梯度。

### 2.3 `nn_applygradient.py`

新增了 `RMSPropNesterov` 分支。对于每一层参数，程序执行：

```python
r = rho * r + (1 - rho) * gradient ** 2
velocity = alpha * velocity - learning_rate * gradient / (sqrt(r) + stability_constant)
parameter = parameter + velocity
```

该逻辑同时支持：

- 权重 `W`；
- 偏置 `b`；
- Batch Normalization 参数 `Gamma`；
- Batch Normalization 参数 `Beta`。

## 3. 使用方法

```python
from NN import NN

nn = NN(
    layer=[6, 20, 20, 2],
    active_function='sigmoid',
    output_function='sigmoid',
    batch_size=100,
    learning_rate=0.01,
    optimization_method='RMSPropNesterov',
    rho=0.9,
    alpha=0.9,
    stability_constant=1e-8,
    batch_normalization=1,
    objective_function='Cross Entropy'
)
```

其中 `learning_rate` 对应算法中的全局学习率 \(\epsilon\)。

## 4. 与 `Momentum` 的对比

### 4.1 `Momentum` 的更新方式

项目中原有的 `Momentum` 使用：

```python
v = rho * v + gradient
parameter = parameter - learning_rate * v
```

现有代码中的 `rho` 固定为 `0.1`，在此处表示动量系数；新方法中的 `rho` 表示平方梯度衰减率，动量系数另用 `alpha` 表示，不能直接混用。现有 Momentum 的 `v` 是梯度累积量，新方法的 `v` 是包含学习率和负号的参数位移，因此两种方法的速度状态不能直接复用。

它具有两个特点：

1. 梯度在当前参数 \(\theta\) 处计算；
2. 所有参数使用同一个学习率缩放，梯度大小没有逐参数归一化。

### 4.2 `RMSPropNesterov` 的更新方式

新方法使用：

```python
lookahead_parameter = parameter + alpha * velocity
r = rho * r + (1 - rho) * gradient ** 2
velocity = alpha * velocity - learning_rate * gradient / (sqrt(r) + stability_constant)
parameter = parameter + velocity
```

它具有两个额外机制：

1. **Nesterov 前瞻梯度**：在预计下一位置计算梯度；
2. **RMSProp 自适应缩放**：根据每个参数近期梯度平方的平均值调整步长。

### 4.3 对比表

| 对比项目 | Momentum | RMSPropNesterov |
|---|---|---|
| 梯度计算位置 | 当前参数 \(\theta\) | 前瞻参数 \(\theta+\alpha v\) |
| 梯度缩放 | 不缩放 | 按参数使用 RMSProp 缩放 |
| 参数步长 | 主要由统一学习率决定 | 每个参数有自适应步长 |
| 保存状态 | 速度 `v` | 速度 `v` 和平方梯度平均 `r` |
| 对不同梯度尺度的适应性 | 较弱 | 较强 |
| 内存开销 | 较小 | 较大 |
| 实现复杂度 | 较低 | 较高 |
| 对学习率敏感性 | 较高 | 通常较低，但仍需调参 |
| 计算开销 | 较低 | 略高，且需要前瞻参数处理 |

## 5. RMSPropNesterov 的优势

### 5.1 对不同参数使用不同步长

RMSProp 根据每个参数的历史梯度平方调整更新量。梯度长期较大的参数会自动减小步长，梯度较小的参数可以保持相对较大的步长。这通常比普通 Momentum 更适合不同层或不同参数尺度差异明显的网络。

### 5.2 改善狭长损失曲面的优化

在狭长、方向尺度差异明显的损失曲面中，普通 Momentum 可能在某些方向震荡。RMSProp 的逐参数缩放可以减弱这种现象。

### 5.3 Nesterov 梯度具有一定的提前修正能力

Nesterov 方法先观察动量可能带来的位置，再计算梯度，因此通常能够更早发现更新方向偏差，在接近最优点时减少过冲。

### 5.4 适合小批量和非平稳梯度

RMSProp 只保留近期梯度平方的指数滑动平均，不像 AdaGrad 那样持续累加并导致学习率不断变小，因此适合小批量训练和梯度统计随训练过程变化的情况。

## 6. RMSPropNesterov 的劣势

### 6.1 内存占用更大

Momentum 主要保存速度；RMSPropNesterov 还要为每个参数保存平方梯度平均值，因此优化器状态更多。

### 6.2 实现和调参更复杂

至少需要选择 `learning_rate`、`rho` 和 `alpha`。这些参数相互影响，错误组合可能导致训练过慢、震荡或不稳定。

### 6.3 前瞻计算增加实现开销

每个小批量都要临时构造前瞻参数，并在梯度计算后恢复原参数。虽然没有额外进行一次完整前向传播，但代码路径和状态管理比普通 Momentum 更复杂。

### 6.4 仍可能受到异常梯度影响

RMSProp 能缓解梯度尺度差异，但不能替代梯度裁剪、合适的初始化或合理的学习率设置。梯度出现 NaN 或极端异常值时，训练仍可能失败。

### 6.5 不保证所有任务都优于 Momentum

在简单、条件良好的问题上，Momentum 的计算更简单、状态更少，可能已经足够，并且可能达到相近甚至更好的结果。最终效果仍应通过验证集性能和训练曲线判断。

## 7. 推荐初始参数

可以从以下参数开始：

```python
learning_rate=0.001
rho=0.9
alpha=0.9
stability_constant=1e-8
```

如果训练震荡，可以降低 `learning_rate` 或 `alpha`；如果训练速度过慢，可以适当提高 `learning_rate`。如果梯度变化非常剧烈，可以尝试提高 `rho`，例如使用 `0.95` 或 `0.99`。

## 8. 验证结果

已对修改后的代码进行以下检查：

- `NN.py`、`nn_train.py`、`nn_applygradient.py` 通过 Python 语法检查；
- 使用一个小型全连接网络完成了 `RMSPropNesterov` 小批量训练；
- 样例中检查的权重和权重 RMSProp 累计量均保持有限数值；
- Batch Normalization 分支包含 Gamma/Beta 的对应更新。

## 9. 实现边界与比较注意事项

上述验证是小型样例验证，没有完成 MNIST 或 Chess 的完整训练，也没有测量两种优化器的准确率或收敛速度。文中的优势和劣势是机制上的分析，不代表本项目已有实测结论；第 7 节参数仅用于试验起点。

本次沿用现有前向和反向传播，未修正原有 Batch Normalization 统计与导数实现，也未修改损失函数。例如，现有交叉熵损失包含 `0.5` 系数，而对应 softmax 反向梯度没有该系数。因此，本次实现保证优化器按上述顺序使用项目返回的梯度，不能据此证明原有所有损失与梯度完全一致。

项目的反向传播已经对 W/b 梯度按实际小批量大小取平均，并包含权重衰减项；这些梯度也会在前瞻参数处计算。小批量划分沿用原逻辑，不能整除批大小的尾部样本仍会被跳过。

做公平比较时，应保持网络结构、数据划分、随机种子和训练预算一致，分别调节学习率，并记录验证准确率、损失、耗时和内存。原有 Momentum 动量系数固定为 `0.1`；若要隔离两种机制的影响，需要先使动量系数可配置。原有 BN Momentum 的 `vGamma` 初始值为 `1`，新方法所有速度初值均为零，这也是比较时需要考虑的差异。

使用新方法应创建对应的新网络；加载原有优化器的检查点后仅修改 `optimization_method` 不会初始化新方法所需的状态。`nn_applygradient` 的新分支要求梯度已经在前瞻参数处计算，建议通过 `nn_train` 使用，直接调用普通 forward/backward/applygradient 流程不会自动获得 Nesterov 前瞻行为。
