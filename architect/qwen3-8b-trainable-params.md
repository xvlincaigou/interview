# 如何根据 `Qwen/Qwen3-8B` 的 `config.json` 正确计算可训练参数量

目标文件：[`https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json`](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json)

这份说明默认讨论的是 `Qwen3ForCausalLM` 做全参数训练（full fine-tuning）时的可训练参数量，也就是所有 `nn.Parameter` 都参与训练的情况。

结论先写在前面：

- `Qwen3ForCausalLM` 的可训练参数量约为 `8,190,735,360`，也就是约 `8.19B`
- 如果你只算 `Qwen3Model` 主干，不算 `lm_head`，则是 `7,568,405,504`

## 1. 先从 config 里读出关键字段

根据 Hugging Face 上的 `config.json`，这个模型的关键配置是：

- `vocab_size = 151936`
- `hidden_size = 4096`
- `intermediate_size = 12288`
- `num_hidden_layers = 36`
- `num_attention_heads = 32`
- `num_key_value_heads = 8`
- `head_dim = 128`
- `attention_bias = false`
- `tie_word_embeddings = false`

这里有两个特别关键的点：

- `num_key_value_heads = 8`，说明它不是标准 MHA，而是 GQA；因此 `K` 和 `V` 的投影维度要按 `8 * 128 = 1024` 来算，而不是按 `32 * 128 = 4096`
- `tie_word_embeddings = false`，说明输入 embedding 和输出 `lm_head` 不是共享权重，因此两者都要单独计入

参考：

- `config.json`：[`Qwen/Qwen3-8B/config.json`](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json)
- Hugging Face `Qwen3` 实现：[`modeling_qwen3.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py)

## 2. 明确模型里到底有哪些可训练模块

按 Hugging Face 的实现，`Qwen3ForCausalLM` 的主要可训练参数来自：

1. `embed_tokens`
2. `36` 个 decoder layer
3. 最终的 `norm`
4. `lm_head`

其中每个 decoder layer 又包含：

1. `self_attn`
2. `mlp`
3. `input_layernorm`
4. `post_attention_layernorm`

而 `self_attn` 里面具体有：

1. `q_proj`
2. `k_proj`
3. `v_proj`
4. `o_proj`
5. `q_norm`
6. `k_norm`

注意：

- `attention_bias = false`，所以 attention 里的线性层都没有 bias
- `Qwen3MLP` 在 Hugging Face 实现里也是 `bias=False`
- Rotary Embedding 不带可训练参数
- `use_cache`、`torch_dtype`、`sliding_window` 这些配置不会增加可训练参数量

## 3. 逐项计算参数量

### 3.1 Token embedding

`embed_tokens = nn.Embedding(vocab_size, hidden_size)`

所以：

`151936 * 4096 = 622,329,856`

## 3.2 每一层 Attention

### `q_proj`

输出维度是：

`num_attention_heads * head_dim = 32 * 128 = 4096`

参数量：

`4096 * 4096 = 16,777,216`

### `k_proj`

输出维度是：

`num_key_value_heads * head_dim = 8 * 128 = 1024`

参数量：

`4096 * 1024 = 4,194,304`

### `v_proj`

和 `k_proj` 一样：

`4096 * 1024 = 4,194,304`

### `o_proj`

输入维度是：

`num_attention_heads * head_dim = 4096`

输出维度是：

`hidden_size = 4096`

参数量：

`4096 * 4096 = 16,777,216`

### `q_norm` 和 `k_norm`

Hugging Face 的实现里：

- `q_norm = Qwen3RMSNorm(head_dim)`
- `k_norm = Qwen3RMSNorm(head_dim)`

每个 RMSNorm 只有一个 `weight` 参数，大小是 `head_dim = 128`

所以：

- `q_norm = 128`
- `k_norm = 128`

### 每层 Attention 总计

`16,777,216 + 4,194,304 + 4,194,304 + 16,777,216 + 128 + 128 = 41,943,296`

## 3.3 每一层 MLP

Hugging Face 实现里：

- `gate_proj: 4096 -> 12288`
- `up_proj: 4096 -> 12288`
- `down_proj: 12288 -> 4096`

因此：

- `gate_proj = 4096 * 12288 = 50,331,648`
- `up_proj = 4096 * 12288 = 50,331,648`
- `down_proj = 12288 * 4096 = 50,331,648`

每层 MLP 总计：

`50,331,648 * 3 = 150,994,944`

## 3.4 每一层的两个 RMSNorm

每个 decoder layer 还有：

- `input_layernorm = Qwen3RMSNorm(hidden_size)`
- `post_attention_layernorm = Qwen3RMSNorm(hidden_size)`

每个参数量都是：

`4096`

所以总计：

`4096 + 4096 = 8,192`

## 3.5 每层总参数量

每层总计：

`Attention + MLP + 2 * LayerNorm`

`41,943,296 + 150,994,944 + 8,192 = 192,946,432`

## 3.6 36 层总参数量

`192,946,432 * 36 = 6,946,071,552`

## 3.7 最终输出前的 RMSNorm

`Qwen3Model` 末尾还有一个：

- `norm = Qwen3RMSNorm(hidden_size)`

参数量：

`4096`

## 3.8 `lm_head`

`Qwen3ForCausalLM` 里有：

- `lm_head = nn.Linear(hidden_size, vocab_size, bias=False)`

所以参数量：

`4096 * 151936 = 622,329,856`

因为 `tie_word_embeddings = false`，这部分不能和 `embed_tokens` 合并，只能单独计入。

## 4. 最终求和

### 4.1 只算主干 `Qwen3Model`

主干包括：

- `embed_tokens`
- `36` 层 decoder
- 最终 `norm`

所以：

`622,329,856 + 6,946,071,552 + 4,096 = 7,568,405,504`

### 4.2 算完整的 `Qwen3ForCausalLM`

再加上 `lm_head`：

`7,568,405,504 + 622,329,856 = 8,190,735,360`

所以完整模型的可训练参数量是：

- `8,190,735,360`
- 约等于 `8.19B`

## 5. 为什么很多人会算错

最常见的错误有这几个：

1. 把 `K/V` 也按 `32` 个头去算

这里其实是 GQA，`num_key_value_heads = 8`，所以 `k_proj` 和 `v_proj` 的输出维度是 `8 * 128 = 1024`，不是 `4096`。

2. 忘记 `q_norm` 和 `k_norm`

Qwen3 的 attention 里除了线性层，还有两个按 `head_dim` 建的 RMSNorm，虽然量不大，但严格计算时要加上。

3. 把 `embed_tokens` 和 `lm_head` 当成共享权重

这里只有当 `tie_word_embeddings = true` 时，才能按一份权重算；这个模型的配置是 `false`，所以两份都要保留。

4. 把 RoPE 当成可训练参数

Rotary position embedding 本身不是 `nn.Parameter`，不计入可训练参数量。

5. 混淆“模型总参数量”和“当前训练参数量”

如果你做的是 LoRA、QLoRA、Adapter tuning，真正训练的参数只是一小部分，不等于这里算出来的 `8.19B`。这里算的是“全参数训练时，模型本身有多少可训练参数”。

## 6. 一个可复核的 Python 公式

```python
vocab_size = 151936
hidden_size = 4096
intermediate_size = 12288
num_hidden_layers = 36
num_attention_heads = 32
num_key_value_heads = 8
head_dim = 128

embed_tokens = vocab_size * hidden_size

q_proj = hidden_size * (num_attention_heads * head_dim)
k_proj = hidden_size * (num_key_value_heads * head_dim)
v_proj = hidden_size * (num_key_value_heads * head_dim)
o_proj = (num_attention_heads * head_dim) * hidden_size
q_norm = head_dim
k_norm = head_dim
attn_per_layer = q_proj + k_proj + v_proj + o_proj + q_norm + k_norm

gate_proj = hidden_size * intermediate_size
up_proj = hidden_size * intermediate_size
down_proj = intermediate_size * hidden_size
mlp_per_layer = gate_proj + up_proj + down_proj

norms_per_layer = hidden_size + hidden_size

layer_total = attn_per_layer + mlp_per_layer + norms_per_layer
all_layers = layer_total * num_hidden_layers

final_norm = hidden_size
lm_head = hidden_size * vocab_size

qwen3_model_params = embed_tokens + all_layers + final_norm
qwen3_for_causal_lm_params = qwen3_model_params + lm_head

print(qwen3_model_params)          # 7568405504
print(qwen3_for_causal_lm_params)  # 8190735360
```

## 7. 一句话总结

如果严格按照 Hugging Face 当前的 `config.json` 和 `modeling_qwen3.py` 来算，`Qwen/Qwen3-8B` 的完整 `Qwen3ForCausalLM` 可训练参数量应为：

`8,190,735,360`，约 `8.19B`。
