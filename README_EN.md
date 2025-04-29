# VLLM-Nexus-Gui

## Program Name
vllm-nexus-gui-vram.py (This version uses only VRAM mode, suitable for users with multiple large-capacity VRAM NVIDIA GPUs)
vllm-nexus-gui-hybrid.py (This version supports hybrid memory mode with both RAM and VRAM)


VLLM is an excellent platform, better than Ollama, with faster speed, but native VLLM doesn't support hybrid memory model deployment.

Besides adding more GPUs, another solution is to use a combination of VRAM and system RAM, sometimes called Unified Memory Management. However, NVIDIA's actual unified memory has high hardware requirements that most computers can't meet. To achieve true unified memory, you would need a new computer with at least DDR5 memory or higher.

This project was developed on DDR3 memory, focusing first on making large models run, then addressing performance optimization.

There are several existing software solutions for running large models in hybrid memory mode, each with their own characteristics. The features of this software include:
1. Loading the large model completely into system memory first, then from memory to VRAM, avoiding errors that might occur when loading directly to VRAM.
2. Dynamic optimization during runtime after memory loading is complete.
3. Graphical user interface, eliminating complex command-line operations.
4. Cross-platform support for both Windows and Ubuntu.

# VLLM-Nexus-Gui: VLLM Server Manager (Memory-Optimized Version)

A high-performance server management system (with GUI) based on vLLM, optimized for large language models, providing intelligent memory management, multi-GPU support, and VRAM optimization. Especially suitable for deploying large models in resource-constrained environments.




## Core Features

- 🚀 Intelligent memory management and CPU offloading
- 💾 Model memory swapping to overcome VRAM limitations
- 🖥️ Multi-GPU tensor parallel computing (supports 1-4 GPU configurations)
- 📊 Real-time GPU and system memory monitoring
- ⚙️ Intelligent parameter recommendation system
- 🔄 Support for models in various precision formats
- 🛠️ Compatible with different versions of VLLM command-line parameters

## Memory Optimization Highlights

- **Memory Swapping Technology**: Allows loading models larger than GPU VRAM
- **Intelligent Memory Pre-allocation**: Reduces memory fragmentation, optimizes large model loading
- **Real-time System Resource Monitoring**: Dynamically adjusts parameters to avoid OOM errors
- **CPU Offloading Mechanism**: Uses system memory as a cache for model weights

## Interface Guide

### Basic Configuration Area
- **Model Path**: Select local model folder
- **IP Address/Port**: Set server listening address
- **GPU Count**: Configure number of GPUs for inference
- **VRAM Ratio**: Control memory usage ratio for each GPU (0.0-1.0)
- **Maximum Tokens**: Set maximum token count in batch processing
- **Maximum Sequence Length**: Supported maximum context window size

### KV Cache Configuration
- **Cache Precision**: Select numerical type for KV cache (float16/float32)
- **Block Size**: Define token count for each cache block
- **Maximum Block Count**: Limit maximum blocks allocated per GPU
- **Dynamic Scaling**: Enable scaling optimization between different batches

### Memory Optimization Settings
- **CPU Offload Size**: Set size of model data offloaded to CPU memory (GB)
- **Memory Swap Space**: Configure disk swap space size (GB)
- **Force Immediate Execution**: Avoid memory shortages caused by CUDA graph capture
- **Memory Buffer Pre-allocation**: Pre-allocate memory to reduce fragmentation

## Memory and VRAM Calculation Guide

### Model Size Estimation

| Model Parameters | FP16 Size | INT8 Size | Minimum GPU Requirement | Optimal GPU Configuration |
|-----------------|-----------|-----------|-------------------------|---------------------------|
| 7B              | ~14GB     | ~7GB      | 16GB                    | 24GB single card          |
| 13B             | ~26GB     | ~13GB     | 24GB×2                  | 32GB×1                    |
| 32B             | ~64GB     | ~32GB     | 40GB×2                  | 80GB×1                    |
| 70B             | ~140GB    | ~70GB     | 80GB×2                  | 80GB×4                    |

### VRAM Usage Breakdown

For a 32B model (FP16), VRAM allocation is approximately:

```
Model weights: 64GB
KV cache (2048 context): ~2GB
Optimizer states: Not applicable for inference
Gradients: Not applicable for inference
Activation values: ~1GB
CUDA kernels: ~0.5GB
--------------------------
Total: ~67.5GB
```

### Memory Swapping and CPU Offloading Calculation

When using memory swapping, you can calculate required resources using this formula:

```
Required GPU VRAM = Model Size × (1 - CPU Offload Ratio) × (1 - VRAM Ratio/100)
Required System RAM = Model Size × CPU Offload Ratio + Buffer (~2GB)
Recommended Swap Space = Model Size × 0.2 (approximately 20% reserved space)
```

For example, loading a 70B model (FP16) on an RTX 4090 (24GB):
```
CPU Offload: ~100GB
GPU VRAM: ~21GB (model processing portion)
System RAM: ~120GB
Swap Space: ~28GB
```

## Quick Start Guide

```bash
# Create virtual environment
python -m venv vvvip
vvvip\Scripts\activate

# Activate virtual environment (Linux/Mac)
source vvvip/bin/activate

# Install dependencies
pip install -r requirements.txt

sudo apt-get install python3-tk

# Launch the program
python vllm-nexus-gui-hybrid.py
```

## Recommended Configuration Plans

### Consumer GPUs (RTX 4090)
- Maximum Model: 13B (full FP16)
- VRAM Ratio: 0.85
- CPU Offloading: Must be enabled for larger models
- Recommended Settings: Use the "Recommended Settings" feature in the interface

### Professional GPUs (A100-80GB)
- Maximum Model: 70B (single card FP16)
- VRAM Ratio: 0.9
- Memory Swapping: Optional, for extra-long contexts
- KV Cache: float16 preferred

### Multi-GPU Configuration (RTX 4090 × 2)
- Maximum Model: 35B (tensor parallel)
- GPU Count: 2
- VRAM Ratio: 0.8
- KV Cache Block Size: 16

## Advanced Usage Tips

1. **Large Model Loading**
   - Enable "Force Immediate Execution" to avoid memory shortages during CUDA graph capture
   - Use lower VRAM ratio (0.75-0.85) to reserve system space

2. **Memory Optimization**
   - Enable "Memory Buffer Pre-allocation" for large models to reduce fragmentation
   - System RAM should be at least twice the model size

3. **Performance Balance**
   - Increasing "Block Size" can reduce cache management overhead
   - Reducing "Maximum Sequence Length" can decrease memory usage per request

## Troubleshooting Common Issues

| Problem | Solution |
|---------|----------|
| CUDA OOM Error | 1. Lower VRAM ratio 2. Enable CPU offloading 3. Use "Recommended Settings" |
| Model Loading Failure | Check if model path contains complete weight files |
| Server Startup Failure | Try using "Alternative Startup Method" to launch the server |
| KV Cache Overflow | Reduce "Maximum Tokens" or increase "Maximum Block Count" |
| Insufficient System Memory | Enable disk swap space or reduce CPU offloading ratio |


---

*Note: This version is specially optimized for memory, focusing on loading ultra-large models beyond VRAM limitations. Parameter settings in the interface directly affect model loading and inference performance, please configure carefully according to your system specifications.*
