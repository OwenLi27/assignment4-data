# CS336 Spring 2026 Assignment 4: Data

For a full description of the assignment, see the assignment handout at
[cs336_assignment4_data.pdf](./cs336_assignment4_data.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

This directory is organized as follows:

- [`./cs336_basics`](./cs336_basics): This module contains the staff 
  implementation of the language model from assignment 1. You will use this training code
  to train an LM on your filtered data. You should not modify the training logic, since
  your leaderboard submission must use it exactly.
- [`./cs336_data`](./cs336_data): This folder is basically empty! This is the
  module where you will implement code to filter and process the data.

Visually, it should look something like:

``` sh
.
├── cs336_basics  # A python module named cs336_basics
│   └── ... an optimized training implementation ...
├── cs336_data  # TODO(you): code that you'll write for assignment 4
│   ├── __init__.py
│   └── ... TODO(you): any other files or folders you need for assignment 4 ...
├── README.md
├── pyproject.toml
└── ... TODO(you): other files or folders you need for assignment 4 ...
```

As in previous assignments, we use `uv` to manage dependencies.

## Downloading data

### 本地自学：统一准备入口

在 assignment4-data 目录运行：

```sh
uv run scripts/prepare_raw_data.py
```

这个命令准备本作业各节需要的原始资源，以及第四节的英文 WET 起点，不需要课程 Modal 权限。

| 阶段 | 内容 | 本地位置 |
| --- | --- | --- |
| assets | 示例 WARC/WET、语言识别/NSFW/仇恨言论模型、Paloma 验证集；补足指定数量的 WET | `local-shared-data/CC/`、`classifiers/`、验证集 `.bin` |
| wiki | Wikipedia XML 分片及其引用 URL 清单 | `local-shared-data/wiki/` |
| english | 用 lid.176.bin 保留英文概率 ≥ 0.7 的文档，保持原文和来源 | `local-shared-data/english-wet-data/` |

默认至少准备 **1 个 Wikipedia 分片和 2 个原始 WET 文件**，已有更多分片时保留并复用。
原始 WET 从 `local-shared-data/` 和其 `CC/` 子目录发现；英文输出和缓存不会被重复当作输入。
这是小规模学习数据，不等于课程完整共享语料。Wikipedia 默认日期与 handout 一致，为 `20260501`；
若官方不再提供该日期，使用 `--wiki-date YYYYMMDD` 指定仍可下载且已完成的 dump。
额外 WET 默认取自 `CC-MAIN-2026-17`，可通过 `--crawl-id` 指定其他 crawl。

重复执行同一命令会校验 `local-shared-data/raw-data-manifest.json`，跳过输入、配置和输出均未变化的阶段。
Wikipedia 分片核对官方 SHA1；其他资源记录本地 SHA256，用于发现后续变化，不等同于官方校验和。
下载成功后才发布正式文件；基础资源的中断下载会重新下载，Wikipedia 分片支持断点续传。
首次接管旧的基础资源时会登记已有文件的哈希，因此应使用可信、已完整下载的文件。

常用选项：

```sh
# 仅准备基础资源，或仅重新检查英文 WET 阶段
uv run scripts/prepare_raw_data.py --steps assets
uv run scripts/prepare_raw_data.py --steps english

# 提高目标数量；这是总量下限，不是每次额外下载的数量
uv run scripts/prepare_raw_data.py --wiki-shards 4 --wet-files 10

# 显式准备全部 Wikipedia 分片，WET 仍按 --wet-files 控制
uv run scripts/prepare_raw_data.py --steps wiki --all-wiki-shards
```

英文阶段会回读新产物，检查英文筛选、文档数量和原文完整性。
后续读取语料时只匹配 `english-wet-data/*.warc.wet.gz`，不要将目录中的 JSON 清单当作 WET。
Wiki URL 清单的实际路径见 `wiki/manifest.json` 的 `url_file` 字段；子集文件名带 `-subset`。

准备到这里即达到第四节的 starting point。引用网页抓取、质量分类训练集构建、模型训练、
第四节的质量过滤/去重以及 GPT-2 分词不在这个入口中；Paloma 文件只用于验证。
现有的 `prepare_wiki_references.py`、`prepare_english_wet.py` 等单步脚本仍可独立使用。

### 单独准备原始资源

```sh
# 仅下载示例、预训练分类器与 Paloma；兼容原来的 --offline-only 命令
uv run scripts/download_data.py --offline-only

# Wikipedia：只处理已完整下载的分片，或额外下载一个分片
uv run scripts/prepare_wiki_references.py --existing-only
uv run scripts/prepare_wiki_references.py --additional-shards 1

# 仅筛选指定的本地 WET 文件
uv run scripts/prepare_english_wet.py --inputs local-shared-data/CC/example.warc.wet.gz
```

日常准备优先使用 `prepare_raw_data.py`；单步脚本用于独立操作。
旧的 Modal/Wikipedia 重复下载流程已移除，英文判定保留在 `cs336_data/wet_files.py`，
文件筛选和验证由 `prepare_english_wet.py` 完成。

### 质量分类器训练数据

以下步骤独立于原始资源准备入口。先准备 Wikipedia 引用清单和 WARC，再抓取正例、提取负例和导出。
示例导出到单独目录，保留当前已训练模型使用的数据集：

```sh
uv run scripts/prepare_quality_data.py negatives --warc local-shared-data/CC/example.warc.gz
uv run scripts/prepare_quality_data.py crawl --target 6500 --workers 4
uv run scripts/export_quality_dataset.py export --output-dir local-shared-data/quality/dataset-rebuilt
uv run scripts/export_quality_dataset.py audit --output-dir local-shared-data/quality/dataset-rebuilt
```

抓取成功率决定最终可用数量；导出时若数据不足，按提示补充来源或调整 `--per-class`。
上述 audit 做数据检查，不训练模型。质量分类器的设计、实验记录和最佳参数见
[数据准备与训练笔记](docs/quality-classifier-train.md)。

### 本地数据与 Git

`local-shared-data/` 中的原始数据、抓取缓存、训练集、实验结果和模型仅保存在本机，不纳入 Git。
`output/`、`wandb/`、测试报告和提交压缩包也已忽略。
`quality_classify.py` 仍使用本机保存的最佳模型；Git 克隆本身不包含该模型，需要另行复制或按笔记训练。

## Training on Modal

The Modal entrypoint in [`./scripts/train.py`](./scripts/train.py) contains the full training config; pass the path to your GPT-2-tokenized training data with `--train-bin`.

The final training run uses 8 B200 GPUs:

```sh
uv run modal run scripts/train.py --train-bin /root/data/your_data.bin
```

## Submitting

To submit, run `./test_and_make_submission.sh` . This script will install your
code's dependencies, run tests, and create a zip with the output. We
should be able to unzip your submitted tarball and run
`./test_and_make_submission.sh` to verify your test results.
