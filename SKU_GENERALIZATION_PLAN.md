# 新 SKU 泛化识别方案

本文档描述后续面向“新增 SKU 识别”的预期方案。目标不是简单换一个检测模型，而是把“商品计数”和“SKU 识别”拆开处理，使系统在新增商品时不必频繁重新训练主模型。

## 目标

当前系统需要解决两个问题：

- 已知商品的可见数量和遮挡数量统计。
- 新增 SKU 能够快速接入识别，不需要每次新增 SKU 都重新训练完整检测模型。

预期目标：

- 新增 SKU 只需要提供少量参考图即可进入识别流程。
- 识别不确定时输出 `unknown_sku` 或 `ambiguous_sku`，避免强行错分到已有 SKU。
- 遮挡商品不单纯依赖置信度，而结合上下文、聚类和展示区域判断是否参与计数。
- 主计数逻辑保持稳定，SKU 识别能力可以独立扩展。

## 核心思路

系统拆成两层：

```text
第一层：可计数商品检测/分割/计数
第二层：SKU 识别与新增 SKU 泛化
```

第一层只判断“哪些实例是需要计数的商品”，不强依赖具体 SKU 类别。

第二层对商品 crop 或 mask 区域做 SKU 匹配，可以通过少样本图库、图像 embedding 或开放词表模型实现。

## 推荐整体流程

```text
输入图片
  -> 商品检测/分割模型
  -> mask 几何分析和聚类
  -> 上下文感知决策：confirmed / confirmed_by_context / unknown / filtered
  -> 商品 crop 或 mask crop
  -> SKU 特征检索
  -> 输出 SKU、置信度和计数结果
```

其中：

- 检测/分割模型负责定位和数量。
- SKU 检索模型负责判断具体是哪一个 SKU。
- 低置信但有上下文支持的遮挡商品，不直接过滤。
- 不像任何 SKU 的候选，输出 `unknown_sku`。

## 决策分层

不要使用一个全局置信度阈值。建议将检测结果分成四类：

### confirmed

高置信无遮挡或清晰商品。

初始规则：

```text
confidence >= 0.80
```

这类结果可以直接参与计数和 SKU 识别。

### confirmed_by_context

置信度不高，但有遮挡或聚类上下文支持。

初始规则：

```text
0.25 <= confidence < 0.80
```

并且满足至少一个上下文条件：

- 与高置信商品在同一挂钩或同一竖向簇中。
- 位于商品展示 ROI 内。
- 形状符合挂装商品特征，例如竖向、细长。
- 与相邻商品排列规律一致。
- 属于某个商品簇中的遮挡部分。

这类结果应参与计数，但可在输出中标记为上下文确认。

### unknown

置信度中等，但缺少上下文支持。

初始规则：

```text
0.25 <= confidence < 0.80
```

并且：

- 不在明确商品簇中。
- 没有高置信商品支持。
- 形状或位置不够确定。

这类结果进入人工复核或 SKU 检索，不应直接作为确定 SKU 输出。

### filtered

明显误检或低置信无上下文支持目标。

过滤条件包括：

- 横向招牌。
- 纸箱、地面、台面或展示架外区域。
- 形状明显不像挂装商品。
- `confidence < 0.25` 且没有任何上下文支持。

## SKU 识别方案

推荐使用少样本 SKU 图像检索，而不是让检测模型直接分类所有 SKU。

### SKU 图库

每个 SKU 建立一个参考图目录：

```text
sku_gallery/
  JM001_九牧增压花洒/
    1.jpg
    2.jpg
    3.jpg
  JM013_九牧健康编织软管/
    1.jpg
    2.jpg
  NEW001_新SKU名称/
    1.jpg
    2.jpg
```

新增 SKU 时，只需要加入 1 到 5 张参考图，然后重新生成特征索引。

### 特征检索

可选特征模型：

- DINOv2
- CLIP
- SigLIP
- 其他适合商品图像检索的 embedding 模型

推理时：

```text
商品 crop -> 提取 embedding -> 与 SKU 图库 embedding 比对 -> 返回 top-k SKU
```

推荐输出：

```json
{
  "sku_id": "JM013",
  "sku_name": "九牧健康编织软管",
  "sku_score": 0.86,
  "sku_source": "gallery_retrieval"
}
```

如果相似度不足：

```json
{
  "sku_id": "unknown_sku",
  "sku_score": 0.52,
  "need_review": true
}
```

如果 top1 和 top2 差距太小：

```json
{
  "sku_id": "ambiguous_sku",
  "top_candidates": [
    {"sku_id": "JM010", "score": 0.78},
    {"sku_id": "JM013", "score": 0.75}
  ],
  "need_review": true
}
```

### 初始阈值建议

阈值需要用现场图片调参，初始可以使用：

```text
top1_score >= 0.75 且 top1_score - top2_score >= 0.08
  -> confirmed_sku

top1_score >= 0.60 但 top1_score - top2_score < 0.08
  -> ambiguous_sku

top1_score < 0.60
  -> unknown_sku
```

## 遮挡商品的 SKU 继承

严重遮挡商品不一定能单独识别 SKU。对于同一挂钩或同一商品簇，可以采用前排可见商品的 SKU 作为簇内遮挡商品的 SKU。

示例：

```json
{
  "cluster_id": 3,
  "sku_id": "JM013",
  "sku_name": "九牧健康编织软管",
  "visible_count": 3,
  "estimated_total": 5,
  "sku_source": "front_visible_item",
  "occlusion_inferred": 2
}
```

这种做法比强行识别每个遮挡实例更稳定。

## 开放词表模型的角色

开放词表或零样本模型可以作为辅助层，而不是最终计数层。

可选用途：

- 发现新商品候选区域。
- 标记可能的 `unknown_sku`。
- 辅助生成训练数据。
- 结合 SAM 生成新 SKU 的候选 mask。

推荐组合：

```text
主计数模型
  + SKU 图像检索
  + 开放词表模型发现新商品
  + 人工复核
  + 回流训练数据
```

不建议完全依赖开放词表模型做遮挡计数，因为遮挡数量通常需要结合展示结构、聚类和商品排列规律。

## 输出字段建议

每个实例建议输出：

```json
{
  "instance_id": 1,
  "decision": "confirmed_by_context",
  "decision_reason": [
    "same_cluster_as_high_conf_instance",
    "vertical_product_shape",
    "inside_display_roi"
  ],
  "class_name": "countable_product",
  "confidence": 0.42,
  "sku_id": "JM013",
  "sku_name": "九牧健康编织软管",
  "sku_score": 0.81,
  "bbox": [100, 200, 180, 560],
  "polygon": []
}
```

整图汇总建议输出：

```json
{
  "confirmed_count": 6,
  "confirmed_by_context_count": 2,
  "unknown_count": 1,
  "filtered_count": 3,
  "estimated_total": 8
}
```

## 分阶段落地计划

### 阶段一：上下文感知后处理

目标：减少单纯置信度阈值带来的误删和误保留。

工作内容：

- 增加 `decision` 字段。
- 增加 `decision_reason` 字段。
- 用高置信实例、聚类、ROI、几何形态共同判断实例状态。
- 保留当前横向招牌过滤。

### 阶段二：单类可计数商品模型

目标：让检测模型专注于“是否需要计数”，而不是直接分类 SKU。

工作内容：

- 将现有 14 类先合并成 `countable_product`。
- 加入大量无关商品、纸箱、展架、地面等负样本。
- 对无关商品图片使用空标签训练。
- 输出稳定的商品 mask 和数量。

### 阶段三：SKU 图库和检索

目标：支持新增 SKU 少样本识别。

工作内容：

- 建立 `sku_gallery/`。
- 生成 SKU embedding 索引。
- 对检测 crop 做 top-k 检索。
- 输出 `confirmed_sku`、`ambiguous_sku`、`unknown_sku`。

### 阶段四：新 SKU 数据闭环

目标：让新增 SKU 从人工确认逐步变成自动识别。

工作内容：

- 将 `unknown_sku` 和 `ambiguous_sku` 保存到复核目录。
- 人工确认后加入 SKU 图库。
- 高质量样本定期回流训练集。
- 周期性重训主计数模型或单类模型。

## 风险和注意事项

- 少样本检索依赖 crop 质量，遮挡、反光、模糊会降低 SKU 识别稳定性。
- 相似包装 SKU 需要更多参考图和更严格的 top1/top2 差距阈值。
- 检测模型误检仍然会发生，需要 ROI、几何规则和 `unknown_sku` 共同兜底。
- 对遮挡严重的实例，优先采用同簇 SKU 继承，不强行单独识别。
- 所有阈值都需要基于现场图片统计后再固定。

## 结论

推荐最终方案：

```text
单类可计数商品分割模型
  + 上下文感知计数
  + SKU 少样本图像检索
  + unknown/ambiguous 拒识机制
  + 人工复核和数据回流
```

这样可以兼顾：

- 遮挡场景下的数量稳定性。
- 新增 SKU 的快速接入。
- 不确定样本的安全拒识。
- 长期数据闭环和模型持续改进。
