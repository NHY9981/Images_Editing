# 深度伪造检测项目

本项目旨在检测深度伪造(Deepfake)图像，包含两种不同的检测方法：自主研发模型(`our_model`)和基于开源模型`VisionRush`的实现。

## 项目结构
```
├── our_model/ # 自主研发模型
│ ├── network/ # 模型网络结构代码
│ ├── preprocess/ # 数据预处理与增强代码
│ └── train.py /# 训练代码，结合了PGD对抗样本训练
└── VisionRush/ # 开源模型DeepFakeDefenders (仅包含主要代码)
```


## 方法说明

### 1. 自主研发模型 (our_model)

首先尝试了自主研发的深度伪造检测模型，包含以下组件：

- **network/**: 自定义神经网络架构
  - 基于EfficientNetV2的深度伪造检测模型
  - Xception网络与Vision Transformer结合，用于深度伪造检测任务
  - M2TR模型：结合RGB和频域特征+多尺度transfomers
  - efficientnet+Vit

- **preprocess/**: 数据预处理与增强
  - 实现了针对真实图片的数据增强技术
  - 通过旋转、翻转、色彩调整等方式扩充真实样本
  - 有效缓解了训练数据中真假样本不平衡的问题

- **train.py**: 训练
  - 结合了对抗训练
  - 使用WeightedRandomSampler进行类别平衡采样

**评估结果**: 
研发模型在测试集上只达到了46%的准确率，效果不好并且训练不稳定。

### 2. 开源模型 VisionRush/DeepFakeDefenders

由于自主研发模型效果未达预期，转而采用了开源的[DeepFakeDefenders](https://github.com/VisionRush/DeepFakeDefenders)模型实现，该模型结合了：

- ConvNeXt架构
- RepLKNet模块
- 采用多域融合和多模型集成的方法

**改进工作**:
1. 将原来研发模型中的数据增强技术应用于此开源模型
2. 调整了模型超参数以适应我们的数据集

**评估结果**:
经过改进后，模型在测试集上的准确率提升至55%，显著优于我们的初始模型。

## 使用说明

## our_model 使用说明

#### 环境配置

```bash
# 安装依赖
pip install -r requirements.txt
```
#### 数据准备

参考[DFGC_starterkit](https://github.com/bomb2peng/DFGC_starterkit/tree/master)
* Run `python preprocess/crop_face.py` to process training set to cropped faces.  
* Run `python preprocess/make_csv.py` to produce train-val split files.

#### 数据增强

```bash
python our_model/preprocess/data_augment.py 
```
具体请修改里面的路径

#### 运行代码
```bash
python our_model/train.py 
```

### VisionRush/DeepFakeDefenders 使用说明

#### 下载预训练权重并调整超参数

```python
RepLKNet---cfg.network.name = 'replknet'; cfg.train.batch_size = 16
ConvNeXt---cfg.network.name = 'convnext'; cfg.train.batch_size = 24
```
####  启动训练
##### 单机多卡训练：（8卡）
```shell
bash main.sh
```
##### 单机单卡训练：
```shell
CUDA_VISIBLE_DEVICES=0 python main_train_single_gpu.py
```
注：此时要删去``mengine.py``243行多余的一个module

#### 模型融合
在``merge.py``中更改ConvNeXt模型路径以及RepLKNet模型路径，执行``python merge.py``后获取最终推理测试模型。

#### 推理

首先运行``infer_api.py``启动api端口，再执行``test.py``测试

## Acknowledgement
Thanks [VisionRush](https://github.com/VisionRush/DeepFakeDefenders) for releasing their powerful model that makes this happen. 


