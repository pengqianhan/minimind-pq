# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MiniMind is a lightweight language model implementation designed for educational purposes and resource-constrained environments. The project provides a complete pipeline from tokenizer training to deployment, including pre-training, supervised fine-tuning (SFT), direct preference optimization (DPO), and model distillation.

## Architecture

**Core Components:**
- `model/`: Contains model architectures and configurations
  - `model_minimind.py`: Main MiniMind transformer implementation with RMSNorm, SwiGLU, and RoPE
  - `model_lora.py`: Low-Rank Adaptation (LoRA) implementation for parameter-efficient fine-tuning
  - `LMConfig.py`: Model configuration classes with support for MoE variants

**Model Variants:**
- MiniMind2-Small: 26M parameters (hidden_size=512, num_layers=8)
- MiniMind2: 104M parameters (hidden_size=768, num_layers=16)
- MiniMind2-MoE: 145M parameters with Mixture-of-Experts (hidden_size=640, num_layers=8)

**Training Pipeline:**
- `trainer/`: All training scripts organized by training stage
  - `train_pretrain.py`: Causal language modeling pre-training
  - `train_full_sft.py`: Supervised fine-tuning for conversational abilities
  - `train_dpo.py`: Direct preference optimization for alignment
  - `train_lora.py`: Parameter-efficient fine-tuning with LoRA
  - `train_distillation.py`: Model distillation for knowledge transfer

**Data Processing:**
- `dataset/`: Data loading and preprocessing utilities
  - `lm_dataset.py`: Custom dataset classes for different training stages

**Evaluation & Deployment:**
- `eval_model.py`: Model evaluation with interactive chat interface
- `scripts/`: Deployment and utility scripts
  - `serve_openai_api.py`: OpenAI-compatible API server
  - `web_demo.py`: Streamlit web interface
  - `train_tokenizer.py`: Custom tokenizer training

## Common Development Commands

### Model Training

**Pre-training (Stage 0):**
```bash
# Single GPU
python trainer/train_pretrain.py

# Multi-GPU
torchrun --nproc_per_node=N trainer/train_pretrain.py
```

**Supervised Fine-tuning (Stage 1):**
```bash
# Single GPU
python trainer/train_full_sft.py

# Multi-GPU
torchrun --nproc_per_node=N trainer/train_full_sft.py
```

**Direct Preference Optimization (Stage 2):**
```bash
python trainer/train_dpo.py
```

**LoRA Fine-tuning:**
```bash
python trainer/train_lora.py
```

### Model Evaluation

**Interactive Chat:**
```bash
python eval_model.py
```

**Evaluate Different Models:**
```bash
# Test pretrained model
python eval_model.py --model_mode 0

# Test SFT model
python eval_model.py --model_mode 1

# Test with LoRA adapter
python eval_model.py --lora_name lora_medical --model_mode 1
```

### Model Deployment

**OpenAI-Compatible API Server:**
```bash
python scripts/serve_openai_api.py
```

**Web Demo:**
```bash
python scripts/web_demo.py
```

### Training Monitoring

**With Weights & Biases:**
```bash
python trainer/train_pretrain.py --use_wandb
```

## Development Guidelines

**Model Loading:**
- Use `model_mode` parameter: 0 for pretrained, 1 for SFT, 2 for RLHF
- Models support both native PyTorch checkpoints and HuggingFace format
- LoRA adapters can be loaded on top of base models

**Training Configuration:**
- All training scripts support distributed training with `torchrun`
- Mixed precision training (FP16) is enabled by default for memory efficiency
- Gradient accumulation and learning rate scheduling are built-in

**Data Requirements:**
- Pre-training: Plain text files for causal language modeling
- SFT: Instruction-response pairs in JSON format
- DPO: Preference pairs with chosen/rejected responses

**Hardware Requirements:**
- Minimum: Single GPU with 8GB VRAM for 26M model
- Recommended: Multiple GPUs for larger models and faster training
- The project is designed to be trainable on consumer hardware

**Model Evaluation:**
- Use `eval_model.py` for comprehensive evaluation with multiple test prompts
- Supports both automated evaluation and interactive chat modes
- Streaming output available for real-time response generation

## Integration with External Tools

**lm-evaluation-harness:**
The project includes the lm-evaluation-harness submodule for standardized model evaluation across various benchmarks.

**Model Formats:**
- Native PyTorch checkpoints for training
- HuggingFace transformers format for deployment
- LoRA adapters stored separately and loaded on demand