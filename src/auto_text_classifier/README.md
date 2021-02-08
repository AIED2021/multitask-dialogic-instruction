# Auto Text Classifier
![atc.png](docs/imgs/atc.png)

## 目的

NLP中有很多任务都是文本二分类任务，但目前基本大家有自己的一套，而且指标也不统一。本项目对现有的文本二分类方法进行汇总（ML+DL），并进行统一，统一有四个方面，仓库化、统一化、工程化和智能化。后续会将二分类扩大到多分类。

## 安装

### 方法一、直接安装(不推荐)

安装依赖
>cpu环境

`pip install -r requirements_cpu.txt`

>gpu环境

gpu则稍微麻烦

第一步：
`conda install tensorflow-gpu==1.13.1  cudatoolkit=10.0.130=0`

第二步:

`pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html`

然后

`pip install -r requirements_gpu.txt`

### 方法二、虚拟环境安装（推荐）

以下以`atc_gpu`举例

1、安装虚拟环境
```sh
conda create --name=atc_gpu python=3.7.5
source activate atc_gpu
```

>确保当前环境是`atc_gpu`

2、安装依赖

第一步：
`conda install tensorflow-gpu==1.13.1  cudatoolkit=10.0.130=0`

第二步:

`pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html`

然后

`pip install -r requirements_gpu.txt`

3、为jupyter安装core(可选）

`ipython kernel install --user --name=atc_gpu`

这时你发现jupyter 多了一个叫`atc_gpu`的core。

删除核心

`jupyter kernelspec remove atc_gpu`

## 使用方式

### 非常重要(必须看!!!)

我们的预训练模型和词向量非常大无法传git，你可以选择：

1、使用`/share/small_project/auto_text_classifier`导入

2、或者手动把`/share/small_project/auto_text_classifier/atc/data` 复制到你自己clone的位置`auto_text_classifier/atc/data`下。

3、利用我们的OSS下载。

### 使用指南

[极简版指南](docs/quick_start.md)

[训练一堆模型](docs/quick_start.md#训练一堆模型)

[api文档](docs/api_detail.md)

### 一些例子
在`example`下有一些demo大家可以参考。
- emb_compare.py : 如何使用多种词向量
- data_report_demo.py : 如何优雅的生成数据的报告。
- bert_demo.py : 展示如何使用bert
- bilstm_demo.py : 展示如何使用lilstm
- aml_demo.py : 批量训练模型python版，有可能碰到显存崩的问题。
- aml_eval_demo.py : 同时评估一批模型。
- train_all.sh : 通过shell来训练一批模型，可以解决批量跑模型显存无法释放问题。



### 支持模型列表
>目前支持的模型如下

#### 中文模型(24个)

| AML 模型名            | 类名     | 默认配置                            | 相关论文                                                     | 模型来源 | 支持多分类 |
| --------------------- | -------- | ----------------------------------- | ------------------------------------------------------------ | -------- | ---------- |
| xlnet_base            | XLNet    | xlnet_base_config                   | [参考文献](https://arxiv.org/abs/1906.08237)                 | [模型来源](https://huggingface.co/hfl/chinese-xlnet-base)      | ✅          |
| xlnet_mid             | XLNet    | xlnet_mid_config                    | [参考文献](https://arxiv.org/abs/1906.08237)                 |  [模型来源](https://huggingface.co/hfl/chinese-xlnet-mid)        | ✅          |
| xlnet_large             | XLNet    | xlnet_large_config                    | [参考文献](https://arxiv.org/abs/1906.08237)                 |  [模型来源](https://huggingface.co/clue/xlnet_chinese_large)        | ✅          |
| nezha_base            | NEZHA    | nezha_base_config                   | [参考文献](https://arxiv.org/abs/1909.00204)                 |[模型来源](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-TensorFlow)           | ✅          |
| nezha_base_wwm        | NEZHA    | nezha_base_wwm_config               | [参考文献](https://arxiv.org/abs/1909.00204)                 |[模型来源](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-TensorFlow)           | ✅          |
| nezha_large           | NEZHA    | nezha_large_config                  | [参考文献](https://arxiv.org/abs/1909.00204)                 |[模型来源](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-TensorFlow)           | ✅          |
| nezha_large_wwm       | NEZHA    | nezha_large_wwm_config              | [参考文献](https://arxiv.org/abs/1909.00204)                 |[模型来源](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-TensorFlow)           | ✅          |
| bert_base             | BERT     | bert_base_config                    | [参考文献](https://arxiv.org/abs/1810.04805)                 |[模型来源](https://huggingface.co/bert-base-chinese)         | ✅          |
| bert_base_wwm         | BERT     | bert_base_www_ext_config            | [参考文献](https://arxiv.org/abs/1906.08101)                 |[模型来源](https://huggingface.co/hfl/chinese-bert-wwm-ext)         | ✅          |
| albert_tiny           | ALBERT   | voidful_albert_chinese_tiny_config  | [参考文献](https://arxiv.org/abs/1909.11942)                 |[模型来源](https://huggingface.co/voidful/albert_chinese_tiny)          | ✅          |
| albert_small          | ALBERT   | voidful_albert_chinese_small_config | [参考文献](https://arxiv.org/abs/1909.11942)                 |[模型来源](https://huggingface.co/voidful/albert_chinese_small)         | ✅          |
| roberta               | ROBERTA  | chinese_roberta_wwm_ext_config      | [参考文献](https://arxiv.org/abs/1907.11692)                 |[模型来源](https://huggingface.co/hfl/chinese-roberta-wwm-ext)         | ✅          |
| roberta_wwm_ext_large | ROBERTA  | hfl_chinese_roberta_wwm_ext_large_config   | [参考文献](https://arxiv.org/abs/1907.11692)                 |[模型来源](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)         | ✅          |
| xlm_roberta_base | ROBERTA  | xlm_roberta_base_config   | [参考文献](https://arxiv.org/abs/1907.11692)                 |[模型来源](https://huggingface.co/xlm-roberta-base)         | ✅          |
| xlm_roberta_large | ROBERTA  | xlm_roberta_large_config   | [参考文献](https://arxiv.org/abs/1907.11692)                 |[模型来源](https://huggingface.co/xlm-roberta-large)         | ✅          |
| electra_large          | ELECTRA  | hfl_chinese_electra_large_config     | [参考文献](https://arxiv.org/abs/2003.10555)                 |[模型来源](https://huggingface.co/hfl/chinese-electra-large-discriminator)     | ✅          |
| electra_base          | ELECTRA  | hfl_chinese_electra_base_config     | [参考文献](https://arxiv.org/abs/2003.10555)                 |[模型来源](https://huggingface.co/hfl/chinese-electra-base-discriminator)         | ✅          |
| electra_small         | ELECTRA  | hfl_chinese_electra_small_config    | [参考文献](https://arxiv.org/abs/2003.10555)                 | [模型来源](https://huggingface.co/hfl/chinese-electra-small-discriminator)       | ✅          |
| bilstm                | BiLSTM   | bilstm_base_config                  | [参考文献](https://d2l.ai/chapter_recurrent-modern/bi-rnn.html) |    -      | ✅          |
| fasttext              | FastText | fasttext_base_config                | [参考文献](https://fasttext.cc/)                             |       -   | ✅          |
| dpcnn                 | DPCNN    | dpcnn_base_config                   | [参考文献](http://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf) |   -       | ✅          |
| textcnn               | TextCNN  | textcnn_base_config                 | [参考文献](https://arxiv.org/pdf/1510.03820)                 |     -     | ✅          |
| textrcnn              | TextRCNN | textrcnn_base_config                | [参考文献](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552) |    -      | ✅          |
| gbdt                  | GBDT     | gbdt_config                         | [参考文献](http://jalammar.github.io/illustrated-transformer/) |    -      | ❌          |

我们同时公布了6个数据集，这里可以用来跑测试或者自己玩。[模型对比结果](docs/model_result.md#模型对比)

#### 英文模型（10个）

| AML 模型名            | 类名     | 默认配置                            | 相关论文                                                     | 模型来源 | 支持多分类 |
| --------------------- | -------- | ----------------------------------- | ------------------------------------------------------------ | -------- | ---------- |
| en_bert_base_cased             | BERT     | en_bert_base_cased_config                    | [参考文献](https://arxiv.org/abs/1810.04805)                 |[模型来源](https://huggingface.co/bert-base-cased)         | ✅          |
| en_roberta_base               | ROBERTA  | en_roberta_base_config      | [参考文献](https://arxiv.org/abs/1907.11692)                 |[模型来源](https://huggingface.co/roberta-base)         | ✅          |
| en_roberta_large               | ROBERTA  | en_roberta_large_config      | [参考文献](https://arxiv.org/abs/1907.11692)                 |[模型来源](https://huggingface.co/roberta-large)         | ✅          |
| en_albert_base_v2           | ALBERT   | en_albert_base_v2_config  | [参考文献](https://arxiv.org/abs/1909.11942)                 |[模型来源](https://huggingface.co/albert-base-v2)          | ✅          |
| en_electra_small          | ELECTRA  | en_google_electra_small_config     | [参考文献](https://arxiv.org/abs/2003.10555) |[模型来源](https://huggingface.co/google/electra-small-discriminator)         | ✅          |
| en_electra_base          | ELECTRA  | en_google_electra_base_config     | [参考文献](https://arxiv.org/abs/2003.10555) |[模型来源](https://huggingface.co/google/electra-base-discriminator)         | ✅          |
| en_electra_large          | ELECTRA  | en_google_electra_large_config     | [参考文献](https://arxiv.org/abs/2003.10555) |[模型来源](https://huggingface.co/google/electra-large-discriminator)         | ✅          |
| en_xlnet_base_cased           | XLNet    | en_xlnet_base_cased_config                   | [参考文献](https://arxiv.org/abs/1906.08237)                 | [模型来源](https://huggingface.co/xlnet-base-cased)      | ✅          |
| en_xlnet_large_cased          | XLNet    | en_xlnet_large_cased_config                    | [参考文献](https://arxiv.org/abs/1906.08237)               |  [模型来源](https://huggingface.co/xlnet-large-cased)        | ✅          |
| en_bart_large          | BART    | en_bart_large_config                    | [参考文献](https://arxiv.org/abs/1910.13461)               |  [模型来源](https://huggingface.co/facebook/bart-large)        | ✅          |


### 支持词向量列表

>中文通用推荐"中文腾讯"，建议先缩放以减少占用,参考[词向量缩放](docs/api_detail.md#词向量缩放),ASR文本建议“自研ASR”。

| 语种 | 词向量           | 配置                          | 维度 | token数    | 备注                                       |
| ---- | ---------------- | ----------------------------- | ---- | ------- | ------------------------------------------ |
| 中文 | 中文腾讯高频        | cn_emb_tencent_small_config   | 200  | 200263  | 通用文本推荐                               |
| 中文 | 中文腾讯         | cn_emb_tencent_config         | 200  | 8825620 | 通用文本推荐。建议先缩放（参考缩放词向量） |
| 中文 | 自研ASR          | cn_emb_zy_asr_config          | 200  | 314040  | ASR文本推荐                                |
| 中文 | 百度百科语料     | cn_emb_bdbk_config            | 300  | 635974  |                                            |
| 中文 | 维基百科语料     | cn_emb_wibk_config            | 300  | 352247  |                                            |
| 中文 | 人民日报语料     | cn_emb_rmrb_config            | 300  | 356037  |                                            |
| 中文 | 搜狗输入法词向量 | cn_emb_sgsrf_config           | 400  | 60528   |                                            |
| 中文 | 搜狗新闻字向量   | cn_emb_sg_char_config         | 300  | 4760    |                                            |
| 英文 | GoogleNews       | en_emb_google_new_high_config | 300  | 27594   |                                            |
| 英文 | Wikipedia        | en_emb_wikipedia_config       | 300  | 999994  |                                            |

我们也做了一些对比实验：[词向量对比](docs/model_result.md#词向量对比)


### 时间

看数据量，基本几个小时左右就可以跑完一个数据的所有实验，下面是每组实验耗费的时间。

|          | 时间/小时 | 训练集合大小 | 验证集合大小 |
| -------- | --------- | ------------ | ------------ |
| 动作描写 | 7         | 27440        | 4843         |
| 幽默识别 | 4         | 13136        | 1642         |
| 环境描写 | 13        | 55746        | 10000        |



## 规划

### 第一阶段

* [x] 仓库化收集常见文本二分类算法（ML+DL）
* [x] 收集常用词向量

### 第二阶段：统一化

* [x] 训练方式统一
* [x] 训练数据
* [x] 数据格式统一
* [x] 环境依赖统一
* [x] demo统一
* [x] 数据报告统一
* [x] 增加模型介绍

### 第三阶段：工程化

* [x] 常用的模型提供工程化代码，支持batch和单句，训练好的模型几乎无须改动即可提交工程化。
* [x] 预测耗时单次
* [ ] 预测耗时batch

### 第四阶段：智能化

* [ ] 自动化训练
* [ ] 自动调参数
* [ ] 生成模型report
* [ ] 工程化代码输出

### 第五阶段：多分类支持


# developer_guide

[developer_guide](docs/developer_guide.md)


# 常见问题

Q："ibstdc++.so.6: version `GLIBCXX_3.4.22' not found"

A：
```shell
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install gcc-4.9
sudo apt-get upgrade libstdc++6
```

Q: 新版axer如何在后台跑

A: 如下
```shell
nohup @ python aml_demo.py -d chnsenticorp>train_chnsenticorp.log&
```