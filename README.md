# PestGuard 研究代码（可运行骨架）

本项目按用户提供的 PestGuard 主文与补充文件实现**防御推理核心**，参考 [AntiStyler 官方演示仓库](https://github.com/IdanYankelev/AntiStyler) 的前五层 VGG19 卷积特征与 Gram 矩阵构造。代码为独立实现，不复制其 Notebook。

## 已实现

1. 原图加 10 像素随机边框，固定随机风格参考；冻结 VGG19，仅对图像做一次风格损失梯度上升。
2. 裁掉边框，计算逐像素 `mean_RGB(abs(crop(z1)-x))`；计算**带符号** MAD 分数 `(r-median(r))/(1.4826*MAD+eps)`。
3. 原图与弱变换视图的真实检测输出；逆映射弱视图框；按类进行匈牙利 IoU 匹配。
4. 仅在匹配框交集内取 `max(sqrt(p*p'))` 构建保护图 `C`。
5. `M0 = 1[a > tau0 + lambda*C]`；八邻域小连通域删除、3×3 闭运算、3×3 膨胀。
6. 对最终掩膜的每个连通域，从周围未掩膜环带取各通道中位数填充；最后运行同一检测器。
7. Faster R-CNN、Ultralytics YOLO、HuggingFace DETR 检测器适配器；保存所有中间图和原始浮点图。
8. 101 点插值 AP50、隐藏与制造攻击成功判定工具函数，以及针对核心规则的测试。
9. 按补充材料拟议几何构建补丁支撑域、替换补丁公式和 M-PGD 的 `16/255` 局部范数投影。

## 必须由真实实验提供

- 训练好的检测器 checkpoint 和其类别 ID 映射。
- VGG19 checkpoint：主文写有 PDT 微调，但未给出权重或微调配置；若使用原始 ImageNet 权重，论文表述必须相应更正。
- 准确的数据划分、攻击图、攻击优化代码与日志。补充文件里的多个数值是**拟议配置**，不能据此声称复现论文表格。
- EOT、DPatch、T-SEA 和自适应 BPDA+EOT 的确切目标函数、随机变换分布与优化记录。`pestguard/patches.py` 只处理几何与约束，并不冒充这些完整攻击实现。
- `style_step_size = eta*beta` 的数值；该值取决于 Gram 归一化、VGG 权重及输入规范，不能从 AP 表格反推。示例配置的 `1.0` 仅用于让程序具备显式默认值，使用前必须校准残差分布。
- 当前 VGG 输入采用 ImageNet 均值/标准差归一化，随机参考图采用固定种子的 `[0,1]` 均匀噪声；主文没有给出这两项的确切实现，复现实验时须与作者代码或训练配置核对。Gram 损失按主文的 Frobenius 范数平方求和。

## 安装

```bash
python -m pip install -r requirements.txt
```

当前代码可在 CPU 运行，但 640×640 的 VGG 梯度步骤建议使用 GPU。若使用 YOLO 或 DETR，再分别安装 `ultralytics` 或 `transformers`。不要用随机初始化的模型做论文实验。

## 推理

### Faster R-CNN

`--num-classes` **包含背景类**。例如一个前景类的 torchvision 模型填 `2`，其前景标签通常为 `1`。

```bash
python run_pestguard.py \
  --input /path/to/test/images \
  --output /path/to/results \
  --detector fasterrcnn \
  --detector-checkpoint /path/to/fasterrcnn.pt \
  --num-classes 2 \
  --vgg-checkpoint /path/to/vgg19.pt \
  --config config_proposed.json \
  --device cuda:0
```

### YOLO / DETR

```bash
python run_pestguard.py --input /path/to/image.jpg --output /path/to/results \
  --detector yolo --detector-checkpoint /path/to/best.pt \
  --vgg-checkpoint /path/to/vgg19.pt --config config_proposed.json
```

对 DETR 使用 `--detector detr`，并将 `--detector-checkpoint` 指向含有模型和 image processor 的**本地** HuggingFace 目录。适配器不主动下载权重。

程序保存 `defended.png`、`style_updated_crop.png`、`weak_view.png`、预览热图、`maps.npz` 与 `result.json`。**MAD 原始值在 `maps.npz` 中**；预览 PNG 为便于查看而作的显示归一化，不是 0–1 的算法分数。

## 评估 PDT 的 YOLO 标注

`evaluate_outputs.py` 读取已保存的预测结果和 YOLO `.txt`。无目标图应有空 `.txt` 文件；缺失标注会报错。若检测器是 torchvision Faster R-CNN、YOLO 类别为 `0`，则用 `--label-class-offset 1 --class-ids 1`。YOLO 检测器通常用 `--label-class-offset 0 --class-ids 0`。

```bash
python evaluate_outputs.py \
  --results /path/to/results \
  --labels /path/to/test/labels \
  --class-ids 1 --label-class-offset 1 \
  --prediction-key detections_final \
  --output /path/to/ap50.json
```

评估器采用补充材料**拟议**的 101 点 AP50、0.001 置信度下限和每图最多 300 框。要对照论文表格，先核实原实验的评价器、数据清单和攻击样本。`hiding_success` 与 `creation_success` 是分开的，不会把两者自动合并成主文未说明的 ASR。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试中的微型假检测器只检查程序连接、坐标与边界条件，**不提供实验性能证据**。

## 与 AntiStyler 的关系

AntiStyler 演示 Notebook 使用优化步骤、基于高分位差值的掩膜、形态学处理及遮罩图像。PestGuard 主文要求**单步**风格更新、MAD 校准、跨视图目标保护和局部中位数填充。本实现遵循 PestGuard 公式；它不是 AntiStyler Notebook 的逐行复刻，也没有声称得到主文中任何 AP 或 ASR 数值。

## 参数来源

| 参数 | 默认值 | 来源/状态 |
|---|---:|---|
| 随机边框 | 10 px | 主文明确描述 |
| VGG 风格层 | 前五个卷积 | 主文/AntiStyler 演示所述，准确层选择需与权重核对 |
| `style_step_size` | 1.0 | 占位有效步长，须实测校准 |
| `eps` | 1e-6 | 补充材料拟议 |
| `rho` | 0.5 | 补充材料拟议 |
| 检测置信度下限 | 0.25 | 补充材料拟议 |
| `tau0, lambda` | 3.0, 2.0 | 补充材料拟议 |
| 最小连通域 | 16 px | 补充材料拟议 |
| 环带内外半径 | 3, 9 px | 补充材料拟议 |
| 弱视图缩放、亮度、对比度 | 0.95–1.05 | 补充材料拟议 |

补充材料里的参考研究数据规模、合成表 S9/S10 和未验证训练设置均**未写入本项目作为真实结果**。
