# Occlusion Visible Count

商品遮挡场景下的可见数量识别项目。主流程使用 YOLO segmentation 输出商品实例，再通过后处理规则过滤展示牌、重复框、弱候选和跨类别重复候选，最终输出每个类别的可见数量。

## 当前最佳版本

当前推荐推理配置：

```text
run_tag: axis_cap_crossdup_v4
weights: outputs/occlusion/data_80_20_baseline/baseline_20e/weights/best.pt
data_yaml: data/data.yaml
imgsz: 640
conf: 0.25
iou: 0.50
max_det: 300
device: 0
```

验证集结果：

```text
images = 72
exact = 56 / 72 = 77.78%
MAE = 0.25
bias_pred_minus_gt = 0.083333
max_abs_error = 2
误差 <= 1: 70 / 72 = 97.22%
```

## 关键代码

推理主链路：

```text
api.py                  FastAPI 批量计数接口
infer.py                命令行单图/目录推理
pipeline.py             统一推理流程
mask_analyzer.py        mask 几何分析、过滤、去重、countability 规则
decision_engine.py      confirmed / confirmed_by_context / unknown 判定
fusion_counter.py       cluster 级计数汇总
visualizer.py           可视化输出
config.py               默认路径和推理参数
utils.py                通用 IO / data.yaml 读取
```

训练与数据准备：

```text
train_seg.py                         YOLO segmentation 训练入口
prepare_seg_dataset.py               分割数据准备
label_convert.py                     标签/多边形转换工具
```

验证与调参：

```text
tools/run_validation_inference.py          验证集批量推理
tools/evaluate_visible_count.py            可见数量评估
tools/sweep_inference_params.py            imgsz/conf/iou 网格搜索
```

## 环境依赖

建议 Python 3.8+，GPU 推理需要本机 CUDA/PyTorch 环境可用。

核心依赖：

```text
ultralytics
torch
opencv-python
numpy
Pillow
PyYAML
fastapi
uvicorn
python-multipart
```

示例安装：

```powershell
pip install ultralytics opencv-python numpy Pillow PyYAML fastapi uvicorn python-multipart
```

如果需要 GPU，请按当前 CUDA 版本安装对应 PyTorch。

## 数据与权重

默认验证数据：

```text
data/data.yaml
data/images/val
data/labels/val
```

当前最佳权重：

```text
outputs/occlusion/data_80_20_baseline/baseline_20e/weights/best.pt
```

注意：`data/`、`outputs/`、`*.pt` 通常被 `.gitignore` 忽略。提交代码时不会自动带上数据和权重，需要单独交付或放到约定的模型目录。

## API 批量计数

启动服务：

```powershell
uvicorn api:app --host 0.0.0.0 --port 8001
```

接口：

```text
POST /api/v1/count/batch
```

请求格式：`multipart/form-data`

参数：

```text
images                必填，可上传一张或多张 jpg/jpeg/png
include_instances     可选，默认 false；true 时返回实例明细
include_visualization 可选，默认 false；true 时返回可视化 base64
```

调用示例：

```powershell
curl -X POST "http://127.0.0.1:8001/api/v1/count/batch" `
  -F "images=@test1.jpg" `
  -F "images=@test2.jpg"
```

默认返回：

```json
{
  "code": 200,
  "msg": "success",
  "data": {
    "total_images": 1,
    "results": [
      {
        "filename": "test.jpg",
        "total_count": 6,
        "items": [
          {
            "category": "九牧增压花洒",
            "count": 1
          },
          {
            "category": "九牧安全角阀",
            "count": 4
          }
        ]
      }
    ]
  }
}
```

API 默认读取当前最佳权重：

```text
outputs/occlusion/data_80_20_baseline/baseline_20e/weights/best.pt
```

也可以用环境变量覆盖：

```powershell
$env:OCCLUSION_WEIGHTS="outputs/occlusion/data_80_20_baseline/baseline_20e/weights/best.pt"
$env:OCCLUSION_DATA_YAML="data/data.yaml"
$env:OCCLUSION_DEVICE="0"
$env:OCCLUSION_IMGSZ="640"
$env:OCCLUSION_CONF="0.25"
$env:OCCLUSION_IOU="0.50"
$env:OCCLUSION_MAX_DET="300"
uvicorn api:app --host 0.0.0.0 --port 8001
```

## 命令行推理

单张图片：

```powershell
python infer.py `
  --source path/to/image.jpg `
  --weights outputs/occlusion/data_80_20_baseline/baseline_20e/weights/best.pt `
  --data-yaml data/data.yaml `
  --device 0 `
  --imgsz 640 `
  --conf 0.25 `
  --iou 0.50 `
  --max-det 300 `
  --skip-depth `
  --run-tag demo_single
```

目录推理：

```powershell
python infer.py `
  --source path/to/images `
  --weights outputs/occlusion/data_80_20_baseline/baseline_20e/weights/best.pt `
  --data-yaml data/data.yaml `
  --device 0 `
  --imgsz 640 `
  --conf 0.25 `
  --iou 0.50 `
  --max-det 300 `
  --skip-depth `
  --run-tag demo_batch
```

输出目录：

```text
outputs/occlusion/occlusion_infer/<run_tag>/visualizations
outputs/occlusion/occlusion_infer/<run_tag>/meta/results.json
```

## 验证集复现

当前最佳配置推理：

```powershell
python tools/run_validation_inference.py `
  --run-tag axis_cap_crossdup_v4 `
  --weights outputs/occlusion/data_80_20_baseline/baseline_20e/weights/best.pt `
  --data-yaml data/data.yaml `
  --device 0 `
  --imgsz 640 `
  --conf 0.25 `
  --iou 0.50 `
  --max-det 300
```

评估：

```powershell
python tools/evaluate_visible_count.py `
  --results outputs/occlusion/occlusion_infer/axis_cap_crossdup_v4/meta/results.json `
  --images data/images/val `
  --labels data/labels/val `
  --data-yaml data/data.yaml
```

查看效果：

```text
outputs/occlusion/occlusion_infer/axis_cap_crossdup_v4/visualizations
outputs/occlusion/occlusion_infer/axis_cap_crossdup_v4/meta/visible_count_eval_summary.json
outputs/occlusion/occlusion_infer/axis_cap_crossdup_v4/meta/visible_count_eval.csv
outputs/occlusion/occlusion_infer/axis_cap_crossdup_v4/meta/worst_visible_count_cases.csv
```

## 参数搜索

用于验证不同 `imgsz/conf/iou` 的组合：

```powershell
python tools/sweep_inference_params.py `
  --python python `
  --sweep-tag multiclss_visible_sweep_v1 `
  --weights outputs/occlusion/data_80_20_baseline/baseline_20e/weights/best.pt `
  --data-yaml data/data.yaml `
  --device 0 `
  --imgsz-values 640,960,1280 `
  --conf-values 0.15,0.20,0.25 `
  --iou-values 0.50,0.60 `
  --max-det 300
```

搜索结果：

```text
outputs/occlusion/param_sweeps/<sweep_tag>/sweep_results.csv
outputs/occlusion/param_sweeps/<sweep_tag>/sweep_results.json
```

## 训练

多类 YOLO segmentation 训练入口：

```powershell
python train_seg.py `
  --data-yaml data/data.yaml `
  --weights yolo11m-seg.pt `
  --epochs 300 `
  --imgsz 896 `
  --batch 8 `
  --device 0 `
  --project runs/occlusion_seg `
  --name yolov11m_seg_multiclass `
  --run-tag multiclass_train
```

训练产物会整理到：

```text
outputs/occlusion/occlusion/<run_tag>/weights/best.pt
outputs/occlusion/occlusion/<run_tag>/weights/last.pt
outputs/occlusion/occlusion/<run_tag>/logs
outputs/occlusion/occlusion/<run_tag>/visualizations
outputs/occlusion/occlusion/<run_tag>/meta/summary.json
```

## 当前后处理核心规则

主要在 `mask_analyzer.py`：

```text
1. 过滤顶部大面积横向展示牌
2. 低置信 context-only 候选过滤
3. 同类强包含重复碎片过滤
4. 同类 axis-only unknown 数量上限
5. 跨类别强重合重复框过滤
6. context/unknown 高包含且中心接近的重复框过滤
```

这些规则对应当前 `axis_cap_crossdup_v4` 最佳结果。

## 提交流程建议

提交代码时建议包含：

```text
api.py
config.py
decision_engine.py
fusion_counter.py
infer.py
mask_analyzer.py
pipeline.py
train_seg.py
visualizer.py
utils.py
label_convert.py
prepare_seg_dataset.py
tools/
README.md
```

不要把下面内容直接提交到代码仓库，除非仓库明确允许大文件：

```text
data/
outputs/
*.pt
*.log
```

权重和数据建议通过网盘、制品库或模型目录单独交付。
