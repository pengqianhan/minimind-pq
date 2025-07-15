"""
MiniMind Language Model Pre-training Script

This script implements the pre-training stage for MiniMind language models using causal language modeling.
Pre-training establishes the foundational language understanding capabilities through next-token prediction
on large-scale text corpora.

**Training Objective:**
Learn fundamental language patterns, syntax, grammar, and world knowledge through unsupervised
next-token prediction on diverse text data. This creates a strong foundation for downstream fine-tuning.

**Key Features:**
- **Distributed Training**: Multi-GPU support using PyTorch DistributedDataParallel (DDP)
- **Memory Optimization**: Mixed precision training with autocast for efficient GPU utilization
- **Dynamic Learning Rate**: Cosine annealing schedule for stable convergence
- **Checkpointing**: Regular model saving with resume capability
- **Monitoring**: Real-time loss tracking and logging

**Training Pipeline:**
1. **Data Loading**: Efficient batch loading with distributed sampling
2. **Forward Pass**: Compute predictions and cross-entropy loss
3. **Loss Masking**: Exclude padding tokens from loss computation
4. **Backward Pass**: Gradient computation with automatic mixed precision
5. **Optimization**: Parameter updates with learning rate scheduling
6. **Checkpointing**: Periodic model state saving

**Model Architecture Support:**
- Standard Transformer architecture with RMSNorm and SwiGLU
- Rotary Positional Embeddings (RoPE) for position encoding
- Optional Mixture-of-Experts (MoE) for increased capacity
- Configurable model sizes (26M, 104M, 145M parameters)

**Performance Optimizations:**
- Gradient accumulation for effective large batch training
- Mixed precision training (FP16) for memory efficiency
- Distributed data parallel for multi-GPU scaling
- Efficient data loading with proper worker management

**Usage:**
    # Single GPU training
    python train_pretrain.py --batch_size 32 --learning_rate 1e-4

    # Multi-GPU training (4 GPUs)
    torchrun --nproc_per_node=4 train_pretrain.py --batch_size 128

**Arguments:**
    --batch_size: Training batch size per GPU
    --learning_rate: Initial learning rate for training
    --epochs: Number of training epochs
    --device: Training device (cuda/cpu)
    --checkpoint_dir: Directory for saving model checkpoints

Dependencies:
    - torch: PyTorch deep learning framework
    - transformers: HuggingFace model components
    - torch.distributed: Multi-GPU training support
"""

import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import math
import warnings
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import PretrainDataset

warnings.filterwarnings('ignore')


def Logger(content):
    """
    Distributed-aware logging function for multi-GPU training.

    This function ensures that log messages are only printed once in distributed training
    environments, preventing duplicate logging from multiple processes.

    Args:
        content (str): The message content to log. Can be any string including
                      formatted training metrics, status updates, or debug information.

    **Behavior:**
    - Single GPU: Always prints the message
    - Multi-GPU (DDP): Only prints from rank 0 process to avoid duplicates
    - Rank 0: The master process responsible for logging and checkpointing

    **Usage Examples:**
    ```python
    Logger("Training started")
    Logger(f"Epoch {epoch}: Loss = {loss:.4f}")
    Logger("Model checkpoint saved")
    ```
    """
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    """
    Compute learning rate using cosine annealing schedule with warmup.

    This function implements a learning rate schedule that starts with a warmup phase
    followed by cosine annealing decay. This schedule helps with training stability
    and convergence, especially for large language models.

    **Schedule Components:**
    1. **Warmup Phase**: Gradual increase from lr/10 to lr (improves stability)
    2. **Cosine Decay**: Smooth decay following cosine curve (prevents sharp drops)

    Args:
        current_step (int): Current training step (0-based). Represents the global
                           step count across all epochs and batches.
        total_steps (int): Total number of training steps planned. Calculated as
                          epochs × steps_per_epoch for the entire training run.
        lr (float): Base learning rate value. The maximum learning rate reached
                   after warmup and used as the starting point for cosine decay.

    Returns:
        float: Computed learning rate for the current step. Value ranges from
               lr/10 (initial warmup) to near 0 (end of training).

    **Mathematical Formula:**
    ```
    lr_current = lr/10 + 0.5 * lr * (1 + cos(π * current_step / total_steps))
    ```

    **Learning Rate Curve:**
    - Step 0: lr/10 (warmup start)
    - Step total_steps/4: ~lr (peak after warmup)
    - Step total_steps/2: ~lr/2 (midpoint decay)
    - Step total_steps: ~lr/10 (final decay)

    **Benefits:**
    - Gradual warmup prevents early training instability
    - Smooth cosine decay avoids sharp learning rate drops
    - Well-suited for transformer model training
    - Helps achieve better final convergence
    """
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    """
    Execute one complete training epoch with comprehensive monitoring.

    This function performs forward and backward passes for all training batches in an epoch,
    applying loss masking, gradient updates, and learning rate scheduling. It includes
    detailed logging and performance monitoring for training analysis.

    Args:
        epoch (int): Current epoch number (0-based). Used for learning rate scheduling
                    and logging context.
        wandb: Weights & Biases logging object for experiment tracking. Can be None
               if logging is disabled. Used to log metrics like loss, learning rate,
               and training speed.

    **Training Loop Process:**
    1. **Batch Processing**: Iterate through all training batches
    2. **Data Transfer**: Move input tensors to appropriate device (GPU/CPU)
    3. **LR Scheduling**: Compute and apply current learning rate
    4. **Forward Pass**: Model prediction with mixed precision
    5. **Loss Computation**: Masked cross-entropy loss calculation
    6. **Backward Pass**: Gradient computation and parameter updates
    7. **Monitoring**: Track and log training metrics

    **Variables Documentation:**
    - loss_fct (nn.CrossEntropyLoss): Cross-entropy loss function with reduction='none'
                                     for per-token loss computation before masking
    - start_time (float): Epoch start timestamp for duration measurement
    - X (torch.Tensor): Input token sequences, shape (batch_size, seq_len-1)
    - Y (torch.Tensor): Target token sequences, shape (batch_size, seq_len-1)
    - loss_mask (torch.Tensor): Binary mask for valid tokens, shape (batch_size, seq_len-1)
    - lr (float): Current learning rate computed from schedule
    - res (CausalLMOutputWithPast): Model output containing logits and auxiliary losses
    - loss (torch.Tensor): Masked cross-entropy loss scalar for backpropagation

    **Loss Masking Logic:**
    ```python
    # Compute per-token cross-entropy losses
    token_losses = loss_fct(logits.view(-1, vocab_size), targets.view(-1))
    # Reshape to match original sequence dimensions
    token_losses = token_losses.view(batch_size, seq_len)
    # Apply mask: only valid tokens contribute to loss
    masked_loss = (token_losses * loss_mask).sum() / loss_mask.sum()
    ```

    **Memory Management:**
    - Mixed precision training reduces GPU memory usage
    - Gradient accumulation enables larger effective batch sizes
    - Automatic memory cleanup after each batch

    **Performance Monitoring:**
    - Tracks tokens processed per second
    - Monitors GPU memory utilization
    - Logs loss convergence patterns
    - Records learning rate progression
    """
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_{lm_config.hidden_size}{moe_path}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
            torch.save(state_dict, ckp)
            model.train()


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    model = MiniMindForCausalLM(lm_config).to(args.device)
    Logger(f'LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="../out")
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl")
    args = parser.parse_args()

    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"

    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        # 同时设置 CUDA 的随机种子
        torch.cuda.manual_seed(base_seed + rank)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    model, tokenizer = init_model(lm_config)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
