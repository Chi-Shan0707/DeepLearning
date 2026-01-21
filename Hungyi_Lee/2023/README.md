我lab1跑完后在runs里头出现了好多奇怪的文件 vscode也打不开，这正常吗<br>

是正常的，那些“奇怪文件”是 TensorBoard 的事件日志。你的代码里用到了 `SummaryWriter()`，训练时会把标量曲线等写到 `runs/` 目录，这些是二进制日志文件（不是文本），VS Code 直接打开会是乱码。

- 什么文件：`runs/` 下的 `events.out.tfevents.*` 和子文件夹，记录损失曲线等。
- 为什么打不开：它们是二进制 protobuf，需用 TensorBoard 查看，不是给编辑器阅读的。

如何查看（WSL 内）
- 启动 TensorBoard：
```bash
conda activate MLenv
tensorboard --logdir runs --host 127.0.0.1 --port 6006
```
- 浏览器打开：`http://127.0.0.1:6006`（Windows/WSL 下也可直接访问）
- VS Code 内查看：Ctrl+Shift+P → “Python: Launch TensorBoard” → 选择 `runs` 目录。

清理与忽略
- 清空日志（安全）： 
```bash
rm -rf runs/*
```
- Git 忽略（建议在仓库根添加 `.gitignore`）：
```
runs/
models/
```

可选设置（减少文件或自定义目录）
- 改日志目录：
```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="runs/lab1")  # 或加日期时间
```
- 暂时禁用记录（减少文件）：把 `writer.add_scalar(...)` 注释掉，或不创建 `SummaryWriter()`。
- 保留最近一次：每次跑前清理旧日志：
```bash
rm -rf runs/lab1 && mkdir -p runs/lab1
```

另外，`models/model.ckpt` 是你保存的最佳模型权重，`pred.csv` 是预测输出，都是正常产物。需要我帮你在项目里加 `.gitignore` 并创建一个更清晰的日志目录（如 `runs/lab1-YYYYMMDD-HHMM`）吗？


***

odels文件夹里.ckpt是什么<br>
简短结论
- `models/*.ckpt` 是训练时保存的“模型检查点”（checkpoint），本质是 PyTorch 的权重文件，用来后续加载做推理或继续训练。它是二进制文件，VS Code 打开会是乱码，属于正常现象。

它里面有什么
- 权重参数：`model.state_dict()` 的张量字典（层名 → 权重/偏置）。
- 你的代码只保存了权重，不包含优化器、学习率等训练状态。

为什么会生成
- 你在 `trainer` 里当验证集损失变好时执行：
  - `torch.save(model.state_dict(), config['save_path'])`
- 所以这个 `.ckpt` 一般是“当前最优”的模型权重快照。

如何加载使用（推理）
```python
import torch
from lab1 import My_Model, COVID19Dataset  # 若不在同文件，请按实际导入

device = 'cpu'  # 若GPU不兼容/不可用，建议用CPU
model = My_Model(input_dim=88).to(device)  # 88取决于你的特征维度
state = torch.load('./models/model.ckpt', map_location=device)
model.load_state_dict(state)
model.eval()

# 例如对 test_loader 做预测（与你现有 predict/test_loader 对齐）
# preds = predict(test_loader, model, device)
```

继续训练可以吗
- 可以，但你只保存了权重，没有保存优化器状态，所以“严格意义的断点续训”不可完全还原。通常你需要：
  - 重新构建优化器，并从较小学习率继续训练，或
  - 修改保存逻辑，连同优化器/epoch等一起保存：
    ```python
    torch.save({
      'epoch': epoch,
      'model': model.state_dict(),
      'optimizer': optimizer.state_dict(),
      'best_loss': best_loss,
    }, './models/model.ckpt')
    ```
  - 加载时再恢复优化器和 epoch。

CPU/GPU 加载注意
- 若之前在 GPU 保存、现在在 CPU 加载：用 `map_location='cpu'`（你现在 GPU 架构不兼容，建议 CPU 推理）。
- 若将来换成支持你显卡的 PyTorch 版本，再加载到 GPU：
  ```python
  state = torch.load('./models/model.ckpt', map_location='cuda')
  model.load_state_dict(state)
  model.to('cuda')
  ```

版本/扩展名说明
- `.ckpt` 只是扩展名习惯，PyTorch 常见 `.pt`、`.pth`；本质相同。
- 建议把 `models/` 放进 `.gitignore`，或用 Git LFS 管理大文件。

需要我给你加上“保存优化器状态的 checkpoint 逻辑”，或写一个“单独推理脚本”只加载 `.ckpt` 生成 `pred.csv` 吗？