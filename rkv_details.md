# R-KV 源码分析与使用指南

本文档基于 `R-KV` 源码进行深入分析，旨在解释其 KV cache 压缩原理，并指导如何在 HuggingFace Transformers 框架下使用该算法。

## 1. 核心原理 (Core Principle)

R-KV 是一种专为长推理（Reasoning）模型（如 DeepSeek-R1）设计的 decoding-time KV cache 压缩算法。传统的 KV 压缩算法（如 SnapKV, H2O）主要针对长 Prompt 优化，而 CoT（Chain-of-Thought）过程产生的长输出具有高度的自相似性和冗余性。

R-KV 的核心思想是在解码过程中，实时评估 Token 的**重要性（Importance）**和**冗余度（Redundancy）**，并保留高价值且低冗余的 Token。

### 1.1 算法流程

R-KV 维护两个主要缓冲区：
1.  **KV Budget (Retained Tokens)**: 长期保留的压缩后 KV cache，固定大小（例如 1024 或 2048）。
2.  **Window Buffer (Recent Tokens)**: 最近生成的 $N$ 个 Token，无条件保留（例如最近 8 个 Token），用于作为“观察者”评估历史 Token 的重要性。

在每步解码时：
1.  **重要性评分 (Importance Scoring)**:
    利用最近的 `window_size` 个 Token 的 Query 向量作为探针，计算其与历史 Key 向量的 Attention 分数。
    $$ \text{Importance} = \text{Pool}(\text{MeanHead}(\text{Softmax}(Q_{\text{recent}} \cdot K_{\text{history}}^T))) $$
    这表示如果一个历史 Token 被最近生成的 Token 频繁关注，它就是重要的。

2.  **冗余度评分 (Redundancy Scoring)**:
    计算历史 Key 向量之间的余弦相似度（Cosine Similarity）。如果一个 Token 与其后续 Token 高度相似，则被视为冗余。
    $$ \text{Redundancy} = \text{Softmax}(\text{MeanHead}(\text{CosSim}(K_{\text{history}}))) $$

3.  **联合筛选 (Joint Selection)**:
    结合两者计算最终得分 $Z$：
    $$ Z = \lambda \cdot \text{Importance} - (1 - \lambda) \cdot \text{Redundancy} $$
    其中 $\lambda$ 是权衡系数。得分最高的 Top-K 个 Token 被保留，其余被驱逐（Evicted）。

## 2. 源码实现分析

相关核心代码位于 `HuggingFace/rkv/` 目录下。

### 2.1 核心逻辑 (`rkv/compression/r1_kv.py`)

`R1KV` 类实现了上述算法。关键方法 `update_kv` 的逻辑如下：

```python
def update_kv(self, key_states, query_states, value_states):
    # 1. 检查是否超过 budget，未超过则不压缩
    if kv_cache_len < self.budget: return key_states, value_states
    
    # 2. 计算 Attention 重要性 (使用最近 window_size 个 query)
    # query_states 实际上这里传入的是 cached_queries
    attn_weights = compute_attention_scores(query_states, key_states)
    attn_cache = ... # Max pooling 处理

    # 3. 计算 Cosine 冗余度
    similarity_cos = cal_similarity(key_states, ...)
    
    # 4. 计算综合得分
    final_score = attn_cache * self.mix_lambda - similarity_cos * (1 - self.mix_lambda)
    
    # 5. 选取 Top-K Token (budget - window_size)
    indices = final_score.topk(self.budget - self.window_size, dim=-1).indices
    
    # 6. 拼接保留的历史 Token 和最近的 Window Token
    # ... gather 操作重组 KV ...
    return compressed_key, compressed_value
```

### 2.2 侵入式集成 (`rkv/monkeypatch.py` & `rkv/modeling.py`)

R-KV 使用 **Monkey Patching** 技术动态替换 Transformers 库中的 `Attention` 模块。

-   **`monkeypatch.py`**: 定义了 `replace_llama`, `replace_qwen2` 等函数。它们将 `transformers.models.llama.modeling_llama.LlamaAttention.forward` 替换为 R-KV 自定义的 `LlamaAttention_forward`。
-   **`modeling.py`**: 定义了自定义的 `forward` 函数。
    -   在 `forward` 中，除了通过 `past_key_value` 维护 KV cache 外，还额外维护了一个 `query_cache`（存储最近 `window_size` 个 Query）。
    -   调用 `self.kv_cluster.update_kv(...)` 执行压缩。
    -   压缩后的 KV 重新写回 `past_key_value`，供后续标准 Attention 计算使用。

## 3. 如何使用 (Usage Guide)

作为算法工程师，要在 HuggingFace 环境下使用 R-KV 压缩您的 LLM，请遵循以下步骤。

### 3.1 安装

首先安装 R-KV 包。假设您已下载源码：

```bash
cd R-KV/HuggingFace  # 注意代码路径可能在 HuggingFace 子目录
pip install -e .
```
*(注：根据目录结构，setup.py 可能在 R-KV 根目录或 HuggingFace 目录下，请根据实际情况运行)*

### 3.2 代码集成示例

使用 R-KV 不需要修改模型权重或重新训练，只需在加载模型前应用 Monkey Patch。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# 导入 R-KV 的 patch 函数
from rkv.monkeypatch import replace_llama, replace_qwen2

# 1. 定义压缩配置
compression_config = {
    "method": "rkv",
    "method_config": {
        "budget": 1024,          # KV Cache 总容量预算 (关键参数)
        "window_size": 8,        # 最近观察窗口大小
        "mix_lambda": 0.07,      # 重要性与冗余度的平衡系数
        "retain_ratio": 0.2,     # 相似度计算中的保留比例
        "retain_direction": "last",
        "first_tokens": 4,       # 始终保留首个 Token (Attention Sink)
    },
    "compression": None,         # 初始状态
    "update_kv": True            # 允许更新 KV
}

model_path = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# 2. 应用 Monkey Patch (必须在 model load 之前)
if "llama" in model_path.lower():
    replace_llama(compression_config)
elif "qwen" in model_path.lower():
    replace_qwen2(compression_config)

# 3. 正常加载模型
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2" # 推荐使用 Flash Attention 2
)
model.eval()

# 4. 正常进行推理
input_text = "Please solve the integral of x^2."
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=4096,
        do_sample=True,
        temperature=0.6
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 3.3 关键参数说明

-   **`budget`**: 最重要的参数。决定了 KV Cache 的最大长度。设置得越小，显存节省越多，但过小可能影响长程推理能力。对于 8B 模型，1024 或 2048 通常是安全的起始值。
-   **`mix_lambda`**: 调节 Importance (Attention) 和 Redundancy (Similarity) 的权重。
    -   $\lambda \to 1$: 更关注 Attention 重要性（类似 H2O）。
    -   $\lambda \to 0$: 更关注去重。
    -   默认值 `0.07` 是论文通过实验得出的经验值，适合数学推理任务。

### 3.4 注意事项

1.  **Model Support**: 目前源码主要支持 Llama 和 Qwen 系列模型。如果使用其他架构（如 Mistral），需要参考 `monkeypatch.py` 和 `modeling.py` 编写相应的 Patch 逻辑。
2.  **Performance overhead**: R-KV 在解码时引入了额外的计算（Attention score 和 Cosine sim）。虽然本身显存占用降低了，但每一步 decoding 会仅仅增加微小的计算延迟。但在长序列生成下，由于 KV 长度被限制在常量 `budget`，整体吞吐量（Throughput）通常会**大幅提升**，且避免了 OOM。

## 4. 扩展至自定义 LLM (Extending to Custom LLMs)

如果您的模型架构不在官方支持列表（目前仅 Llama/Qwen），可以按照以下步骤扩展 R-KV。

### 4.1 理解修改点

R-KV 的核心侵入点在于 Attention 层的 `forward` 函数。您需要：
1.  在 `Attention` 初始化时，注入 `kv_cluster` 对象。
2.  在 `Attention` 前向传播时，拦截 `past_key_value` 的更新逻辑。
3.  维护一个额外的 `query_cache`（用于重要性计算）。

### 4.2 实现步骤

假设您要支持一个新的模型架构 `Mystral`（虚构）。

**第一步：复制并修改 Attention 初始化**

参考 `modeling.py` 中的 `LlamaAttention_init`：

```python
# 原始初始化函数
# def MystralAttention_init(self, config, layer_idx): ...

# 修改后的初始化
def MystralAttention_init(self, config, layer_idx, compression_config):
    # 1. 调用原始初始化 (需根据具体模型调整，这里假设是 nn.Module.__init__)
    super(MystralAttention, self).__init__() 
    # ... (复制原始初始化的其余代码) ...
    
    # 2. 注入压缩模块
    self.config.update(compression_config)
    from rkv.compression import KV_COMPRESSION_MAP
    self.kv_cluster = KV_COMPRESSION_MAP[compression_config["method"]](
        **compression_config["method_config"]
    )
```

**第二步：修改 Attention Forward**

这是最关键的一步。您需要复制原模型的 `forward` 代码，并在 KV Cache 更新前插入压缩逻辑。

```python
def MystralAttention_forward(self, hidden_states, ... , past_key_value=None, ...):
    # ... (前面的 QKV 投影代码保持不变) ...
    query_states = self.q_proj(hidden_states) ...
    key_states = self.k_proj(hidden_states) ...
    value_states = self.v_proj(hidden_states) ...

    # === R-KV 核心逻辑开始 ===
    if past_key_value is not None:
        # 1. 维护 Query Cache (用于 Attention 评分)
        if not hasattr(past_key_value, "query_cache"):
            past_key_value.query_cache = {}
        
        # 存入当前 Query
        if self.layer_idx not in past_key_value.query_cache:
            # Prefill 阶段：取最后 window_size 个
            past_key_value.query_cache[self.layer_idx] = query_states[:, :, -self.config.method_config["window_size"]:, :]
        else:
            # Decoding 阶段：拼接并保持滑动窗口
            past_key_value.query_cache[self.layer_idx] = torch.cat(
                (past_key_value.query_cache[self.layer_idx], query_states), dim=2
            )
            # 截断
            if past_key_value.query_cache[self.layer_idx].shape[-2] > self.config.method_config["window_size"]:
                 past_key_value.query_cache[self.layer_idx] = past_key_value.query_cache[self.layer_idx][:, :, -self.config.method_config["window_size"]:, :]

        # 2. 执行压缩
        cached_queries = past_key_value.query_cache[self.layer_idx]
        # 调用核心压缩算法
        key_states_compress, value_states_compress = self.kv_cluster.update_kv(
             key_states, 
             cached_queries, 
             value_states
        )

        # 3. 将压缩后的 KV 更新到 Cache 中
        # 注意：这里替换了原始的 past_key_value.update 调用
        past_key_value.update(
            key_states_compress, 
            value_states_compress, 
            self.layer_idx, 
            cache_kwargs
        )
    # === R-KV 核心逻辑结束 ===

    # ... (后续计算 Attention 的代码保持不变) ...
    attn_output = ...
    return attn_output, ...
```

**第三步：注册 Monkey Patch**

编写一个新的 patch 函数：

```python
import transformers.models.mystral.modeling_mystral as modeling_mystral

def replace_mystral(compression_config):
    def init_wrapper(self, config, layer_idx):
        OrderedDict_init(self, config, layer_idx, compression_config)

    # 替换类方法
    modeling_mystral.MystralAttention.__init__ = init_wrapper
    modeling_mystral.MystralAttention.forward = MystralAttention_forward
```

### 4.3 调试建议

1.  **Shape Check**: 在 `forward` 中打印 `key_states_compress` 的 shape，确保其长度稳定在 `budget` 附近，没有随序列长度线性增长。
2.  **Verify Output**: 使用短文本进行测试，对比开启 R-KV 前后的输出文本一致性。在 `budget` 充足（如 2048）的情况下，输出应几乎完全一致。
