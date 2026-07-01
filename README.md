# 遮挡商品识别与计数

核心流程是训练 YOLO 分割模型，对单张图片或图片目录做推理，过滤常见的横向招牌误检，并输出商品数量、类别、置信度、边界框和多边形轮廓等结构化结果。

## 功能

- 支持商品自定义类别的 YOLO-seg 训练。
- 支持单图和目录批量推理。
- 支持 FastAPI 接口服务。
- 支持中文类别名输出。
- 输出实例级信息：类别、置信度、bbox、polygon、mask 面积、中心点、主方向角。
- 对上方横向招牌误检做几何后处理过滤。
- 支持可选的 Depth Anything V2 深度估计；也可以使用 `--skip-depth` 跳过深度，只按可见实例计数。

## 项目结构

```text
occlusion/
  api.py                  FastAPI 接口服务
  config.py               默认路径和训练/推理参数
  depth_estimator.py      Depth Anything V2 深度估计封装
  fusion_counter.py       可见数量与遮挡推断数量融合
  infer.py                命令行推理入口
  label_convert.py        bbox、polygon、mask 转换工具
  mask_analyzer.py        mask 几何分析、聚类和误检过滤
  prepare_seg_dataset.py  YOLO bbox 数据集转 YOLO-seg 数据集
  train_seg.py            YOLO-seg 训练入口
  utils.py                图片、YAML、JSON 和目录工具
  visualizer.py           mask、标签和计数结果可视化
```

数据集、模型权重、训练输出和推理输出不会提交到 Git 仓库，需要单独保存或传输。


## 环境安装

建议使用 Python 3.8 及以上版本。基础依赖：

```bash
pip install ultralytics opencv-python numpy pyyaml scikit-learn pillow fastapi uvicorn python-multipart
```

PyTorch 请根据本机 CUDA 或 CPU 环境单独安装。

## 数据集格式

默认使用 YOLO-seg 数据集结构：

```text
data_occlusion/
  data.yaml
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

如果原始标注是 YOLO bbox 格式，可以先转换成 YOLO-seg polygon：

```bash
python -m occlusion.prepare_seg_dataset \
  --src-root /path/to/source_dataset \
  --dst-root /path/to/data_occlusion \
  --splits train val \
  --polygon-mode bbox
```

如果需要用 SAM 细化 polygon，可以使用 `--polygon-mode sam`，并传入 SAM checkpoint。

## 训练

在包含 `occlusion` 包的父目录执行：

```bash
python -m occlusion.train_seg \
  --data-root ./data_occlusion \
  --epochs 200 \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --name occlusion_seg
```

训练产物会整理到：

```text
outputs/occlusion/occlusion/<run_tag>/
  weights/best.pt
  weights/last.pt
  visualizations/
  logs/
  meta/summary.json
```

## 推理

单张图片推理：

```bash
python -m occlusion.infer \
  --source /path/to/image.jpg \
  --weights /path/to/best.pt \
  --data-yaml /path/to/data_occlusion/data.yaml \
  --device 0 \
  --skip-depth
```

目录批量推理：

```bash
python -m occlusion.infer \
  --source /path/to/images \
  --weights /path/to/best.pt \
  --run-tag batch_test \
  --skip-depth
```

推理结果保存到：

```text
outputs/occlusion/occlusion_infer/<run_tag>/
  visualizations/
  meta/results.json
```

`results.json` 主要字段：

- `summary`：总可见数量、估计总数、遮挡推断数量和各聚类计数结果。
- `instances`：参与计数的商品实例。
- `filtered_instances`：被后处理过滤掉的实例。

## 横向招牌误检过滤

部分门店图片中，上方横向招牌可能被模型误识别为商品。当前使用几何规则做后处理过滤，满足以下特征的实例会在计数前被移除：

- mask 面积相对整图较大；
- 中心点位于图片上方区域；
- mask 主方向接近水平。

被过滤的目标不会进入聚类和计数，但会写入 `filtered_instances`，并带有：

```json
"filter_reason": "top_horizontal_display_sign"
```

这样后续可以检查过滤是否合理。

## API 服务

启动服务：

```bash
uvicorn occlusion.api:app --host 0.0.0.0 --port 8001
```

常用环境变量：

```bash
OCCLUSION_WEIGHTS=/path/to/best.pt
OCCLUSION_DEVICE=0
OCCLUSION_IMGSZ=640
OCCLUSION_CONF=0.25
OCCLUSION_IOU=0.5
OCCLUSION_SKIP_DEPTH=true
```

接口：

- `POST /api/v1/occlusion/count`
- `POST /api/v1/occlusion/analyze`
- 兼容接口：
  - `POST /api/occlusion/count`
  - `POST /api/occlusion/analyze`

调用示例：

```bash
curl -X POST "http://127.0.0.1:8001/api/v1/occlusion/analyze" \
  -F "images=@/path/to/image.jpg"
```

## 注意事项

- `data.yaml` 必须使用 UTF-8 编码，避免中文类别名乱码。
- 可视化标签使用 Pillow 和系统中文字体绘制，支持中文显示。
- `__pycache__`、虚拟环境、训练输出、推理输出和模型权重已在 `.gitignore` 中忽略。
- 当前过滤规则是针对“上方横向招牌”的启发式规则，如果后续出现特殊商品形态，需要结合数据继续调整阈值或改为类别/区域约束。
