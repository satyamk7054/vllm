# Multi-LoRA Tuning

**Note**: The LoRA configuration folder should be specified by exporting `VLLM_TUNED_CONFIG_FOLDER=/path/to/configs`.
Without this, the shrink/expand kernels will use default configurations.

## Tuning Process

Multi-lora shrink/expand Triton kernel tuning follows a similar methodology from
[Triton MoE tuning](https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py).

1. Define the searching space. Here is an example of searching space:

   ```python
   block_m_range = [16, 32, 64, 128, 256]
   block_n_range = [32, 64, 128, 256]
   block_k_range = [32, 64, 128, 256]
   num_warps_range = [4, 8]
   num_stage_range = [2, 3, 4, 5]
   num_ctas_range = [1]
   split_k_range = [4, 8, 16, 32, 64]
   ```

2. Get all hidden_state sizes and num_slices that the target model uses for a specific TP size.

   For example, you can acquire the info by simply checking
   [add_lora_linear](https://github.com/vllm-project/vllm/blob/main/vllm/lora/punica_wrapper/punica_gpu.py#L181):

   ```python
   print(f"x_shape: {x.view(-1, x.shape[-1]).shape}")
   print(f"num_slices: {len(output_slices)}")
   for i in range(len(output_slices)):
       print(f"a{i} shape: {lora_a_stacked[i].shape}")
       print(f"b{i} shape: {lora_b_stacked[i].shape}")
   print("y_shape", y.shape)
   ```

3. Benchmark the shrink/expand kernel runtime with different kernel configurations generated from the pre-defined search space
   by performing a grid search to find the optimal kernel configuration.
   vLLM's [benchmark_lora.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_lora.py)
   can be used to search for configurations for different shapes.

4. Tune per `num_active_loras`. The shrink/expand grid's third axis is `num_active_loras` (the number
   of adapters active in a batch), and the optimal config — most sharply `split_k` — depends on it, so
   search separately at each `num_active_loras` you expect in production, using
   `benchmark_lora.py --num-active-loras N`.

## Config Files

### File Naming

| Kernel Type               | File Name Template                          | Example                                      |
| ------------------------- | ------------------------------------------- | -------------------------------------------- |
| shrink                    | `{gpu_name}_SHRINK.json`                    | `NVIDIA_H200_SHRINK.json`                    |
| expand                    | `{gpu_name}_EXPAND_{add_input}.json`        | `NVIDIA_H200_EXPAND_TRUE.json`               |
| fused_moe_lora_w13_shrink | `{gpu_name}_FUSED_MOE_LORA_W13_SHRINK.json` | `NVIDIA_H200_FUSED_MOE_LORA_W13_SHRINK.json` |
| fused_moe_lora_w13_expand | `{gpu_name}_FUSED_MOE_LORA_W13_EXPAND.json` | `NVIDIA_H200_FUSED_MOE_LORA_W13_EXPAND.json` |
| fused_moe_lora_w2_shrink  | `{gpu_name}_FUSED_MOE_LORA_W2_SHRINK.json`  | `NVIDIA_H200_FUSED_MOE_LORA_W2_SHRINK.json`  |
| fused_moe_lora_w2_expand  | `{gpu_name}_FUSED_MOE_LORA_W2_EXPAND.json`  | `NVIDIA_H200_FUSED_MOE_LORA_W2_EXPAND.json`  |

The `gpu_name` can be automatically detected by calling `torch.cuda.get_device_name()`.

### JSON Structure

Config files use the nesting `config_data[primary][num_slices][m][k][n][i]`, where `i` is an optional
dimension in the `fused_moe_lora` configuration representing the intermediate size of the MoE layer.
The shrink/expand ops key by `num_active_loras` (the count the kernel grid runs on); the
`fused_moe_lora_*` ops key by `max_loras`:

```text
config_data[num_active_loras][num_slices][m][k][n][i]   # shrink / expand
config_data[max_loras][num_slices][m][k][n][i]          # fused_moe_lora_*
```

The loader selects the bucket nearest the runtime value, so a file need only contain the buckets you
tuned.

**Backward compatibility.** Shrink/expand configs were previously keyed by `max_loras`; they are now
read as `num_active_loras`. Because selection is nearest-neighbor, a file with a single top-level key
resolves to that key at every active count — identical to the old behavior — so existing tuned folders
keep working unchanged. Supplying a config per `num_active_loras` (multiple top-level keys) is what
opts into per-active-count selection.
