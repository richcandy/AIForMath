# AIForMath

AIForMath 是一个面向 Lean 4 的神经定理证明项目。项目使用 DeepSeek-Prover-V2-7B 作为基础模型，在高中数学 Lean tactic 数据上进行 LoRA 微调，并通过 vLLM 提供 OpenAI-compatible API。最终验证流程在 `MiniF2F_Highschool` 数据集上执行 beam search：模型生成候选 tactic，Lean/Pantograph 并行验证每一步，直到证明闭合或搜索预算耗尽。

## 项目概览

核心组件：

- 基础模型：`deepseek-ai/DeepSeek-Prover-V2-7B`
- 证明语言：Lean 4
- Lean 版本：`leanprover/lean4:v4.29.0-rc1`
- mathlib 版本：`1bc7728a050fc18ca2683f614c531cd7050ff063`
- 训练数据：`Deepseek_highschool_data/`
- 验证数据集：`MiniF2F_Highschool/`
- 训练脚本：`scripts/train_deepseekprover_v2_lean_dojo.py`
- vLLM 服务脚本：`scripts/run_deepseek_prover_with_vllm.sh`
- 最终验证脚本：`evaluation/evaluate_minif2f_highschool_beam_vllm.py`
- TUI 入口：`python -m tui`

默认服务约定：

- vLLM API：`http://127.0.0.1:11451/v1`
- vLLM 模型名：`my-tactic-lora`
- 基础模型目录：`Deepseek-Prover-V2/`
- LoRA adapter 目录：`outputs/deepseekprover_v2_highschool_tactic/`

## 目录结构

```text
.
├── Deepseek-Prover-V2/                         # DeepSeek-Prover-V2-7B 本地模型目录
├── Deepseek_highschool_data/                   # 高中数学 tactic 训练数据
│   ├── train.json
│   ├── val.json
│   ├── test.json
│   └── flatten_data/                           # 训练脚本使用的 messages JSONL 数据
├── MiniF2F_Highschool/                         # 高中数学 MiniF2F Lean 验证集
│   ├── Valid/
│   ├── Test/
│   ├── Valid.lean
│   └── Test.lean
├── evaluation/
│   └── evaluate_minif2f_highschool_beam_vllm.py # 最终验证脚本
├── scripts/
│   ├── flatten_tactic_data_to_jsonl.py          # 将 traced tactic 数据展开为训练 JSONL
│   ├── train_deepseekprover_v2_lean_dojo.py     # DeepSeek-Prover-V2 LoRA 训练脚本
│   ├── run_deepseekprover_v2_tactic_training.sh # 后台启动训练
│   └── run_deepseek_prover_with_vllm.sh         # 启动 vLLM OpenAI API 服务
├── tui/                                        # 交互式 beam search TUI
├── outputs/                                    # 训练输出、LoRA adapter、日志
├── formal_results_storage/                     # 验证结果输出
├── lakefile.lean                               # Lean 项目配置
├── lean-toolchain                              # Lean toolchain 固定版本
└── lake-manifest.json                          # Lean 依赖锁定文件
```

## 硬件与系统要求

推荐环境：

- Linux 服务器
- CUDA GPU
- 至少 2 张可用 GPU 用于当前默认 vLLM 配置
- Conda
- Git
- elan / lake / Lean 4

当前脚本中的 vLLM 配置使用：

```text
--tensor-parallel-size 2
--distributed-executor-backend ray
--gpu-memory-utilization 0.75
--max-model-len 4096
--enable-lora
--max_lora_rank 64
```

如果只有 1 张 GPU，需要修改 `scripts/run_deepseek_prover_with_vllm.sh` 中的：

```bash
--tensor-parallel-size 1
```

并根据显存调整：

```bash
--gpu-memory-utilization 0.75
--max-model-len 4096
```

## Lean 4 环境

项目通过 `lean-toolchain` 固定 Lean 版本：

```text
leanprover/lean4:v4.29.0-rc1
```

`lakefile.lean` 固定 mathlib commit：

```text
1bc7728a050fc18ca2683f614c531cd7050ff063
```

安装 elan 后，在项目根目录构建 Lean 项目：

```bash
lake build
```

如果 `lake` 找不到，请确认 elan 路径已加入环境变量：

```bash
export PATH="$HOME/.elan/bin:$PATH"
```

## Python 环境

当前服务器工作流使用 Conda 环境。进入项目后先激活环境：

```bash
conda activate ty
```

核心 Python 依赖包括：

```text
torch
transformers
datasets
peft
trl
bitsandbytes
openai
pantograph
textual
vllm
ray
```

Flash Attention 是可选加速组件。训练脚本会优先使用 `flash_attention_2`，如果不可用则回退到 `sdpa`。

注意：`pyproject.toml` 当前没有完整声明本项目实际用到的所有依赖，复现时应以这里列出的运行依赖和当前 Conda 环境为准。

## 模型下载

基础模型使用 DeepSeek-Prover-V2-7B。推荐使用 Hugging Face 下载到项目根目录的 `Deepseek-Prover-V2/`：

```bash
huggingface-cli download deepseek-ai/DeepSeek-Prover-V2-7B \
  --local-dir Deepseek-Prover-V2 \
  --local-dir-use-symlinks False
```

如果服务器无法访问 Hugging Face，可以先在本地下载，再上传整个目录到：

```text
AIForMath/Deepseek-Prover-V2/
```

最终目录中应能看到类似文件：

```text
config.json
tokenizer.json
tokenizer_config.json
model.safetensors.index.json
model-00001-of-000002.safetensors
model-00002-of-000002.safetensors
```

`scripts/run_deepseek_prover_with_vllm.sh` 同时兼容 Hugging Face cache 结构。如果 `Deepseek-Prover-V2/refs/main` 存在，脚本会自动解析到 `snapshots/<hash>/`。

## 数据集

### 训练数据

训练数据位于：

```text
Deepseek_highschool_data/
```

原始 split：

```text
Deepseek_highschool_data/train.json
Deepseek_highschool_data/val.json
Deepseek_highschool_data/test.json
```

训练脚本默认读取展开后的 messages JSONL：

```text
Deepseek_highschool_data/flatten_data/train.jsonl
Deepseek_highschool_data/flatten_data/val.jsonl
```

当前 `flatten_data/manifest.json` 中记录的数据规模：

```text
train: 2009 theorems, 6114 tactic step records
val:   1004 theorems, 5427 tactic step records
test:  1007 theorems, 5637 tactic step records
```

每条训练样本是一个 chat messages 记录：

```json
{
  "messages": [
    {"role": "system", "content": "You are a Lean 4 tactic generator..."},
    {"role": "user", "content": "<Lean goal state>"},
    {"role": "assistant", "content": "<one Lean tactic>"}
  ]
}
```

如需重新生成 `flatten_data/`：

```bash
python scripts/flatten_tactic_data_to_jsonl.py \
  --source-root Deepseek_highschool_data \
  --output-root Deepseek_highschool_data/flatten_data
```

### 验证数据

最终验证使用：

```text
MiniF2F_Highschool/
```

目录结构：

```text
MiniF2F_Highschool/Valid/*.lean
MiniF2F_Highschool/Test/*.lean
MiniF2F_Highschool/Valid.lean
MiniF2F_Highschool/Test.lean
```

验证脚本默认读取 `test` split。

## LoRA 训练

训练脚本默认配置：

```text
model root: Deepseek-Prover-V2/
train data: Deepseek_highschool_data/flatten_data/train.jsonl
eval data:  Deepseek_highschool_data/flatten_data/val.jsonl
output dir: outputs/deepseekprover_v2_lean_dojo/
LoRA r:     64
epochs:     2
batch size: 1
grad accum: 16
max length: 2048
```

可以先做一次 dry run，确认数据、模型、trainer 能正常初始化：

```bash
python scripts/train_deepseekprover_v2_lean_dojo.py \
  --train-data-path Deepseek_highschool_data/flatten_data/train.jsonl \
  --eval-data-path Deepseek_highschool_data/flatten_data/val.jsonl \
  --output-dir outputs/deepseekprover_v2_highschool_tactic \
  --dry-run
```

后台启动默认双卡训练：

```bash
bash scripts/run_deepseekprover_v2_tactic_training.sh
```

也可以指定输出目录：

```bash
bash scripts/run_deepseekprover_v2_tactic_training.sh outputs/deepseekprover_v2_highschool_tactic
```

训练脚本会输出日志和 pid 到：

```text
outputs/logs/
```

查看日志：

```bash
tail -f outputs/logs/<train-log>.log
```

训练完成后，vLLM 服务默认使用：

```text
outputs/deepseekprover_v2_highschool_tactic/
```

如果你的 LoRA adapter 在其他目录，需要同步修改 `scripts/run_deepseek_prover_with_vllm.sh` 中的：

```bash
LORA_ADAPTER_PATH="/path/to/adapter"
```

## 启动 vLLM 服务

最终验证和 TUI 都依赖 vLLM OpenAI-compatible API。

启动服务：

```bash
bash scripts/run_deepseek_prover_with_vllm.sh
```

脚本当前硬编码了本机路径：

```bash
BASE_MODEL_PATH="/home/weidu/ty/GraduationProject/AIForMath/Deepseek-Prover-V2"
LORA_ADAPTER_PATH="/home/weidu/ty/GraduationProject/AIForMath/outputs/deepseekprover_v2_highschool_tactic"
/home/weidu/anaconda3/envs/vllm_ty/bin/python -m vllm.entrypoints.openai.api_server
```

在其他机器复现时，必须根据自己的路径修改：

- `BASE_MODEL_PATH`
- `LORA_ADAPTER_PATH`
- vLLM Python 解释器路径
- GPU 数量对应的 `--tensor-parallel-size`

服务启动后检查模型列表：

```bash
curl http://127.0.0.1:11451/v1/models
```

正常情况下应能看到模型名：

```text
my-tactic-lora
```

## MiniF2F_Highschool 验证

最终验证脚本是：

```text
evaluation/evaluate_minif2f_highschool_beam_vllm.py
```

它的流程是：

1. 从 `MiniF2F_Highschool/Test/*.lean` 或 `Valid/*.lean` 读取定理。
2. 将末尾 `:= by sorry` / `:= by admit` 替换为待证明目标。
3. 使用 Pantograph 启动 Lean server，获取当前 goal state。
4. 通过 vLLM 调用 `my-tactic-lora` 生成候选 tactic。
5. 使用 Lean 并行验证 tactic 是否推进或关闭目标。
6. 使用 beam search 继续搜索直到证明成功或预算耗尽。
7. 将逐题结果写入 JSONL，并生成 summary JSON。

### 小样本验证

建议先跑 3 道题确认环境正确：

```bash
python evaluation/evaluate_minif2f_highschool_beam_vllm.py \
  --input-root MiniF2F_Highschool \
  --split test \
  --max-samples 3 \
  --output formal_results_storage/minif2f_highschool_smoke.jsonl \
  --summary-path formal_results_storage/minif2f_highschool_smoke.summary.json \
  --api-base http://127.0.0.1:11451/v1 \
  --model-name my-tactic-lora \
  --beam-width 5 \
  --num-tactics 12 \
  --lean-workers 6
```

### 完整验证

下面的配置对应当前项目中已有完整验证结果使用的一组较强搜索参数：

```bash
python evaluation/evaluate_minif2f_highschool_beam_vllm.py \
  --input-root MiniF2F_Highschool \
  --split test \
  --output formal_results_storage/minif2f_highschool_beam_vllm.jsonl \
  --summary-path formal_results_storage/minif2f_highschool_beam_vllm.summary.json \
  --api-base http://127.0.0.1:11451/v1 \
  --api-key EMPTY \
  --model-name my-tactic-lora \
  --beam-width 10 \
  --max-depth 21 \
  --num-tactics 20 \
  --max-new-tokens 96 \
  --temperature 0.0 \
  --top-p 1.0 \
  --repetition-penalty 1.0 \
  --lean-workers 6 \
  --time-budget-seconds 1000 \
  --max-expanded-nodes 256 \
  --max-verified-tactics 3600 \
  --server-timeout 60 \
  --max-hammers-per-node 2
```

已有结果示例中，`MiniF2F_Highschool` test split 共 89 道题，成功率约为 `0.6629`。实际结果会受模型 checkpoint、GPU、vLLM 版本、搜索参数和 Lean 环境影响。

### 输出格式

逐题 JSONL 输出包含字段：

```text
id
split
file_path
question
theorem_statement
proof
predictions
solved
proof_steps
closed_at_depth
expanded_nodes
verified_tactics
visited_states
failure
elapsed_seconds
generation_wall_time
verification_wall_time
layer_summaries
generation_errors
verification_errors
```

summary JSON 包含：

```text
total_theorems
success_rate
avg_expanded_nodes
avg_verified_tactics
avg_generation_wall_time
avg_verification_wall_time
avg_solution_depth
failure_counts
```

## TUI 界面

TUI 是验证流程的交互式界面，依赖同一个 vLLM 服务和 Lean/Pantograph 验证逻辑。

先启动 vLLM：

```bash
bash scripts/run_deepseek_prover_with_vllm.sh
```

再启动 TUI：

```bash
python -m tui
```

默认配置：

```text
input root: MiniF2F_Highschool
split: test
api base: http://127.0.0.1:11451/v1
model name: my-tactic-lora
beam width: 5
max depth: 21
num tactics: 12
lean workers: 6
time budget: 480 seconds
```

快捷键：

```text
r: run selected problem
s: stop current search
q: quit
```

界面会显示：

- 当前题目
- root goal
- beam 状态
- tactic 候选与验证结果
- 当前 proof steps
- 日志和最终结果

## 常见问题

### vLLM 启动后验证脚本找不到模型

检查服务是否启动：

```bash
curl http://127.0.0.1:11451/v1/models
```

确认输出中有：

```text
my-tactic-lora
```

如果模型名不同，要在验证脚本中使用对应的 `--model-name`。

### Lean 或 Pantograph 报错

先确认 Lean 项目能构建：

```bash
lake build
```

再确认 elan 路径：

```bash
export PATH="$HOME/.elan/bin:$PATH"
```

如果 mathlib 没有下载完成，`lake build` 会先拉取和构建依赖。

### CUDA/NCCL 报错

当前 vLLM 脚本设置了：

```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=DEBUG
```

如果多卡通信仍失败，可以先用单卡降低复杂度：

```bash
--tensor-parallel-size 1
```

### 显存不足

可以尝试降低：

```bash
--gpu-memory-utilization
--max-model-len
--num-tactics
--beam-width
--lean-workers
```

### 完整验证耗时较长

先使用 `--max-samples` 跑小样本。完整 test split 会同时消耗 vLLM 推理资源和多个 Lean worker CPU 进程。

## 复现顺序

从零复现建议按以下顺序执行：

1. 安装 elan、Conda、CUDA/PyTorch/vLLM 等依赖。
2. 下载 `deepseek-ai/DeepSeek-Prover-V2-7B` 到 `Deepseek-Prover-V2/`。
3. 准备 `Deepseek_highschool_data/` 和 `MiniF2F_Highschool/`。
4. 运行 `lake build` 构建 Lean 项目。
5. 如需训练，运行 `scripts/flatten_tactic_data_to_jsonl.py` 和 `scripts/run_deepseekprover_v2_tactic_training.sh`。
6. 启动 `scripts/run_deepseek_prover_with_vllm.sh`。
7. 用 `curl http://127.0.0.1:11451/v1/models` 检查服务。
8. 先跑小样本验证。
9. 再跑完整 `MiniF2F_Highschool` test 验证。
10. 如需交互式查看搜索过程，运行 `python -m tui`。

## Git 注意事项

模型权重、训练输出、验证结果和日志通常很大，不适合直接提交到 Git。提交前建议检查：

```bash
git status
git diff --stat
```

不要提交密钥、token、`.env`、本地模型权重、checkpoint、大型 wheel 文件或日志文件。
