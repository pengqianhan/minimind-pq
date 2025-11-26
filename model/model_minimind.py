# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    r"""
    Configuration class for MiniMind models.

    This class defines all the parameters necessary to build a MiniMind model. It inherits
    from `transformers.PretrainedConfig` and can be used to instantiate a model according
    to the specified configuration.

    Attributes:
        dropout (float, optional, defaults to 0.0): The dropout probability for all fully connected
            layers in the embeddings, encoder, and pooler.
        bos_token_id (int, optional, defaults to 1): The ID of the beginning-of-sequence token.
        eos_token_id (int, optional, defaults to 2): The ID of the end-of-sequence token.
        hidden_act (str, optional, defaults to 'silu'): The non-linear activation function
            (function or string) in the encoder and pooler.
        hidden_size (int, optional, defaults to 512): Dimensionality of the encoder layers and
            the pooler layer.
        intermediate_size (int, optional): Dimensionality of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder. If None, it is calculated based on `hidden_size`.
        max_position_embeddings (int, optional, defaults to 32768): The maximum sequence length
            that this model might ever be used with.
        num_attention_heads (int, optional, defaults to 8): Number of attention heads for each
            attention layer in the Transformer encoder.
        num_hidden_layers (int, optional, defaults to 8): Number of hidden layers in the
            Transformer encoder.
        num_key_value_heads (int, optional, defaults to 2): Number of key and value heads for
            Grouped Query Attention (GQA).
        vocab_size (int, optional, defaults to 6400): Vocabulary size of the MiniMind model.
        rms_norm_eps (float, optional, defaults to 1e-05): The epsilon used by the RMS normalization
            layers.
        rope_theta (float, optional, defaults to 1000000.0): The base period for Rotary
            Positional Embeddings (RoPE).
        flash_attn (bool, optional, defaults to True): Whether to use Flash Attention, a memory-efficient
            attention implementation.
        use_moe (bool, optional, defaults to False): Whether to use a Mixture-of-Experts (MoE)
            architecture for the feed-forward network.
        num_experts_per_tok (int, optional, defaults to 2): The number of "top" experts to route
            each token to in the MoE layers.
        n_routed_experts (int, optional, defaults to 4): The total number of experts in each MoE layer.
        n_shared_experts (int, optional, defaults to 1): The number of shared experts. (Not currently used in this implementation).
        scoring_func (str, optional, defaults to 'softmax'): The function to compute gating scores in the MoE layer.
        aux_loss_alpha (float, optional, defaults to 0.1): The scaling factor for the auxiliary load-balancing
            loss in the MoE layers.
        seq_aux (bool, optional, defaults to True): Whether to compute the auxiliary loss on a sequence
            level.
        norm_topk_prob (bool, optional, defaults to True): Whether to normalize the probabilities of the
            top-k experts in the MoE gating mechanism.
    """
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            inference_rope_scaling: bool = False,
            flash_attn: bool = True,
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.1,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings
        self.rope_scaling = {
            "beta_fast": 4,
            "beta_slow": 1,
            "factor": 4,
            "original_max_position_embeddings": 2048,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn
        # -------------------------------------------------- #
        #       Mixture-of-Experts (MoE) Configuration       #
        # `use_moe` must be True for these to be effective.  #
        # -------------------------------------------------- #
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        """
        Root Mean Square Layer Normalization.

        This layer normalizes the input tensor by the root mean square of its elements,
        scaled by a learnable weight parameter.

        Args:
            dim (int): The dimension of the input tensor to normalize over. This is typically
                       the hidden size of the model.
            eps (float, optional): A small value added to the denominator for numerical
                                   stability. Defaults to 1e-5.
        """
        super().__init__()
        self.eps = eps
        # weight (torch.Tensor): A learnable scaling parameter of shape (dim,).
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Computes the normalization component.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): The input tensor of shape (..., dim).

        Returns:
            torch.Tensor: The normalized and scaled tensor, with the same type as the input.
        """
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    Applies Rotary Positional Embeddings (RoPE) to the query and key tensors.

    RoPE injects positional information by rotating chunks of the query and key tensors
    based on their position.

    Args:
        q (torch.Tensor): The query tensor of shape (bsz, seq_len, num_heads, head_dim).
        k (torch.Tensor): The key tensor of shape (bsz, seq_len, num_heads, head_dim).
        cos (torch.Tensor): The precomputed cosine frequencies from `precompute_freqs_cis`.
                            Shape: (seq_len, head_dim).
        sin (torch.Tensor): The precomputed sine frequencies from `precompute_freqs_cis`.
                            Shape: (seq_len, head_dim).
        position_ids (Optional[torch.Tensor]): Not used in this implementation. Kept for
                                               compatibility. Defaults to None.
        unsqueeze_dim (int, optional): The dimension along which to unsqueeze the cosine and
                                       sine tensors to match the input tensor shapes.
                                       Defaults to 1.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the transformed query and key tensors:
            - q_embed (torch.Tensor): The query tensor with applied RoPE.
            - k_embed (torch.Tensor): The key tensor with applied RoPE.
    """
    def rotate_half(x):
        """Rotates half the hidden dimensions of the input tensor."""
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats the key and value tensors for Grouped Query Attention (GQA).

    In GQA, the number of key/value heads is smaller than the number of query heads.
    This function expands the key and value tensors to match the number of query heads
    by repeating them `n_rep` times. This allows for efficient attention calculation
    while reducing the memory footprint of the K/V cache.

    Args:
        x (torch.Tensor): The input tensor to repeat. Expected shape is
                          (batch_size, seq_len, num_key_value_heads, head_dim).
        n_rep (int): The number of times to repeat the tensor. This is the ratio of
                     query heads to key/value heads.

    Returns:
        torch.Tensor: The expanded tensor with shape
                      (batch_size, seq_len, num_key_value_heads * n_rep, head_dim).
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    Multi-head attention mechanism with Grouped Query Attention (GQA) and optional Flash Attention.

    This module implements the core attention logic for the MiniMind model. It supports:
    - Multi-head attention (MHA) where each head attends to a different part of the input.
    - Grouped Query Attention (GQA) where multiple query heads attend to the same key/value head
      to reduce computational cost and memory usage.
    - Rotary Positional Embeddings (RoPE) for incorporating positional information.
    - An optional, highly optimized Flash Attention implementation for improved performance
      (requires PyTorch >= 2.0).
    - Key-value (KV) caching for efficient generation.

    Attributes:
        n_local_heads (int): The number of query heads.
        n_local_kv_heads (int): The number of key/value heads.
        n_rep (int): The replication factor for GQA (n_local_heads / n_local_kv_heads).
        head_dim (int): The dimension of each attention head.
        q_proj (nn.Linear): Linear layer to project input to query.
        k_proj (nn.Linear): Linear layer to project input to key.
        v_proj (nn.Linear): Linear layer to project input to value.
        o_proj (nn.Linear): Linear layer to project the combined head outputs back to hidden size.
        attn_dropout (nn.Dropout): Dropout layer applied to the attention scores.
        resid_dropout (nn.Dropout): Dropout layer applied to the output projection.
        flash (bool): A flag indicating whether Flash Attention is available and enabled.
    """
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
            position_embeddings (Tuple[torch.Tensor, torch.Tensor]): A tuple containing the
                precomputed cosine and sine frequencies for RoPE.
            past_key_value (Optional[Tuple[torch.Tensor, torch.Tensor]], optional): A tuple containing
                the cached key and value tensors from previous steps, used during auto-regressive
                generation. Defaults to None.
            use_cache (bool, optional): If True, the computed key and value tensors are returned
                for caching. Defaults to False.
            attention_mask (Optional[torch.Tensor], optional): A mask to prevent attention to
                certain positions, typically used for padding. Shape: (batch_size, seq_len).
                Defaults to None.

        Returns:
            Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]: A tuple containing:
                - output (torch.Tensor): The output tensor of the attention module, of shape
                  (batch_size, seq_len, hidden_size).
                - past_kv (Optional[Tuple[torch.Tensor, torch.Tensor]]): The updated key-value
                  cache if `use_cache` is True, otherwise None.
        """
        bsz, seq_len, _ = x.shape
        # Project input to query, key, and value
        xq, xk = self.q_proj(x), self.k_proj(x)
        # Reshape for multi-head attention
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # Apply Rotary Positional Embeddings
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # Key-Value Caching: concatenate past KVs with current KVs
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # Repeat KVs for Grouped Query Attention and transpose for attention calculation
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        if self.flash and seq_len != 1:
            dropout_p = self.dropout if self.training else 0.0
            attn_mask = None
            if attention_mask is not None:
                attn_mask = attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1)
                attn_mask = attn_mask.bool() if attention_mask is not None else None

            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=True)
        else:
            # Standard (manual) attention path
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # Apply causal mask to prevent attending to future tokens
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)

            # Apply padding mask
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        # Combine head outputs and apply final projection
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN).

    This module implements the feed-forward sublayer of a Transformer block. It consists of
    three linear transformations with a SwiGLU activation in between.
    FFN formula: Dropout(down_proj(SwiGLU(gate_proj(x), up_proj(x))))
    where SwiGLU(x, y) = (SiLU(x) * y)

    Attributes:
        gate_proj (nn.Linear): The first linear transformation (part of SwiGLU).
        down_proj (nn.Linear): The final linear transformation which projects back to the hidden size.
        up_proj (nn.Linear): The second linear transformation (part of SwiGLU).
        dropout (nn.Dropout): Dropout layer applied to the output of the final projection.
        act_fn (function): The activation function (e.g., SiLU).
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        # If intermediate_size is not specified, calculate it based on a common heuristic.
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            # Ensure the intermediate size is a multiple of 64 for efficiency.
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Feed-Forward Network.

        Args:
            x (torch.Tensor): The input tensor from the attention sublayer.
                              Shape: (batch_size, seq_len, hidden_size).

        Returns:
            torch.Tensor: The output tensor of the FFN.
                          Shape: (batch_size, seq_len, hidden_size).
        """
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    """
    Gating mechanism for the Mixture-of-Experts (MoE) layer.

    This module determines which experts to route each token to. It takes the hidden states
    as input and outputs routing weights and expert indices. It also calculates an
    auxiliary load-balancing loss to encourage experts to be used equally.

    Attributes:
        top_k (int): The number of experts to route each token to.
        n_routed_experts (int): The total number of available experts.
        scoring_func (str): The function used to calculate routing scores (e.g., 'softmax').
        alpha (float): The coefficient for the auxiliary loss.
        seq_aux (bool): Flag to indicate if auxiliary loss is computed sequence-wise.
        norm_topk_prob (bool): Flag to normalize the top-k probabilities.
        weight (nn.Parameter): The learnable weights for the gating network, of shape
                               (n_routed_experts, hidden_size).
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        """
        Forward pass for the MoE gating mechanism.

        Args:
            hidden_states (torch.Tensor): The input tensor from the previous layer.
                                          Shape: (batch_size, seq_len, hidden_size).

        Returns:
            tuple: A tuple containing:
                - topk_idx (torch.Tensor): The indices of the top-k experts.
                                           Shape: (batch_size * seq_len, top_k).
                - topk_weight (torch.Tensor): The routing weights for the top-k experts.
                                              Shape: (batch_size * seq_len, top_k).
                - aux_loss (torch.Tensor): The auxiliary load-balancing loss. A scalar tensor.
        """
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    """
    Mixture-of-Experts Feed-Forward Network.

    This module replaces the standard FeedForward network with an MoE layer. Instead of a single
    dense network, it uses a gating mechanism to route each token to a small number of "expert"
    feed-forward networks. It also supports shared experts that are used by all tokens.

    Attributes:
        gate (MoEGate): The gating network that selects which experts to use for each token.
        experts (nn.ModuleList): A list of standard `FeedForward` networks (the "routed experts").
        shared_experts (nn.ModuleList, optional): A list of `FeedForward` networks that are applied
                                                   to all tokens, if `n_shared_experts > 0`.
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        """
        Forward pass for the MOEFeedForward network.

        This method uses a two-stage process:
        1. **Training path**: Used when `model.training` is True. It processes tokens for all
           experts simultaneously, which is efficient for training with backpropagation.
        2. **Inference-optimized path (`moe_infer`)**: Used during evaluation (`torch.no_grad`).
           It processes tokens expert by expert to minimize memory usage, which is crucial for
           running large models.

        Args:
            x (torch.Tensor): The input tensor from the attention sublayer.
                              Shape: (batch_size, seq_len, hidden_size).

        Returns:
            torch.Tensor: The output tensor of the MoE layer.
                          Shape: (batch_size, seq_len, hidden_size).
        """
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            # Training path: process all experts in parallel
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=torch.float16)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            # Inference path: memory-optimized
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        # Add output from shared experts, if any
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        Memory-optimized inference path for the MoE layer.

        This function processes the input expert by expert. It sorts the tokens based on their
        assigned expert index, processes all tokens for a single expert in a batch, and then
        scatters the results back to their original positions. This is slower but much more

        Args:
            x (torch.Tensor): The flattened input tensor. Shape: (batch_size * seq_len, hidden_size).
            flat_expert_indices (torch.Tensor): The flattened indices of the chosen expert for each token.
                                                Shape: (batch_size * seq_len * top_k,).
            flat_expert_weights (torch.Tensor): The flattened weights for the chosen expert for each token.
                                                Shape: (batch_size * seq_len * top_k, 1).

        Returns:
            torch.Tensor: The output of the MoE layer. Shape: (batch_size * seq_len, hidden_size).
        """
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok

        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class MiniMindBlock(nn.Module):
    """
    A single Transformer block (layer) of the MiniMind model.

    This module implements one layer of the Transformer architecture, consisting of a multi-head
    self-attention mechanism followed by a position-wise feed-forward network. Each sublayer
    is wrapped with a residual connection and layer normalization (specifically RMSNorm).

    The block follows the standard Transformer architecture:
    1. Input is normalized and passed through self-attention
    2. Residual connection adds the original input to the attention output
    3. The result is normalized and passed through the feed-forward network
    4. Another residual connection adds the pre-FFN state to the FFN output

    Attributes:
        layer_id (int): The index of this layer within the model (0-based).
        num_attention_heads (int): Number of attention heads.
        hidden_size (int): The dimensionality of the hidden states.
        head_dim (int): The dimension of each attention head (hidden_size / num_attention_heads).
        self_attn (Attention): The multi-head self-attention mechanism.
        input_layernorm (RMSNorm): Layer normalization applied before self-attention.
        post_attention_layernorm (RMSNorm): Layer normalization applied before the feed-forward network.
        mlp (Union[FeedForward, MOEFeedForward]): The feed-forward network, which can be either
                                                  a standard FFN or a Mixture-of-Experts FFN.
    """
    def __init__(self, layer_id: int, config: MiniMindConfig):
        """
        Initialize a MiniMind Transformer block.

        Args:
            layer_id (int): The index of this layer within the model (0-based).
            config (MiniMindConfig): Configuration object containing model hyperparameters.
        """
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Choose between standard FFN and MoE FFN based on configuration
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """
        Forward pass through the Transformer block.

        This method implements the standard Transformer layer computation with pre-normalization:
        1. Apply layer norm to input, then self-attention with residual connection
        2. Apply layer norm to result, then feed-forward network with residual connection

        Args:
            hidden_states (torch.Tensor): Input tensor containing the hidden states from the previous layer.
                                          Shape: (batch_size, seq_len, hidden_size).
            position_embeddings (Tuple[torch.Tensor, torch.Tensor]): Precomputed cosine and sine
                                                                     frequencies for RoPE. Each tensor
                                                                     has shape (seq_len, head_dim).
            past_key_value (Optional[Tuple[torch.Tensor, torch.Tensor]]): Cached key and value tensors
                                                                          from previous decoding steps.
                                                                          Used for efficient generation.
                                                                          Defaults to None.
            use_cache (bool): Whether to return updated key-value cache for the next step.
                             Defaults to False.
            attention_mask (Optional[torch.Tensor]): Mask to avoid performing attention on padding
                                                    token indices. Values should be 0 for tokens
                                                    to mask and 1 for tokens to attend to.
                                                    Shape: (batch_size, seq_len). Defaults to None.

        Returns:
            Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]: A tuple containing:
                - hidden_states (torch.Tensor): Output hidden states after processing through
                                               the block. Shape: (batch_size, seq_len, hidden_size).
                - present_key_value (Optional[Tuple[torch.Tensor, torch.Tensor]]): Updated key-value
                                                                                  cache if use_cache
                                                                                  is True, otherwise None.
        """
        # Store input for residual connection
        residual = hidden_states
        
        # Pre-normalization and self-attention with residual connection
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        
        # Pre-normalization and feed-forward network with residual connection
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    """
    The core MiniMind Transformer model.

    This model implements the main Transformer architecture consisting of token embeddings,
    multiple Transformer blocks (layers), and final layer normalization. It processes input
    token IDs through the following pipeline:
    
    1. **Token Embedding**: Converts input token IDs to dense vector representations
    2. **Positional Encoding**: Applies Rotary Positional Embeddings (RoPE) for position awareness
    3. **Transformer Layers**: Processes tokens through multiple self-attention and feed-forward layers
    4. **Final Normalization**: Applies final layer normalization to the output representations
    
    The model supports both training and inference modes, with optimizations for efficient
    key-value caching during autoregressive generation.

    Attributes:
        config (MiniMindConfig): Configuration object containing model hyperparameters.
        vocab_size (int): Size of the vocabulary (number of possible tokens).
        num_hidden_layers (int): Number of Transformer blocks in the model.
        embed_tokens (nn.Embedding): Token embedding layer mapping token IDs to dense vectors.
                                    Shape: (vocab_size, hidden_size).
        dropout (nn.Dropout): Dropout layer applied after token embeddings.
        layers (nn.ModuleList): List of MiniMindBlock instances (Transformer layers).
        norm (RMSNorm): Final layer normalization applied to the output.
        freqs_cos (torch.Tensor): Precomputed cosine frequencies for RoPE. Registered as buffer.
                                  Shape: (max_position_embeddings, head_dim).
        freqs_sin (torch.Tensor): Precomputed sine frequencies for RoPE. Registered as buffer.
                                  Shape: (max_position_embeddings, head_dim).
    """
    def __init__(self, config: MiniMindConfig):
        """
        Initialize the MiniMind model.

        Args:
            config (MiniMindConfig): Configuration object containing all model hyperparameters
                                   including hidden size, number of layers, attention heads, etc.
        """
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        
        # Token embedding layer
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
        # Stack of Transformer blocks
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        
        # Final layer normalization
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, theta=config.rope_theta)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        """
        Forward pass through the MiniMind model.

        This method processes input tokens through the complete Transformer pipeline and returns
        the final hidden states along with optional key-value caches and auxiliary losses.

        Args:
            input_ids (Optional[torch.Tensor]): Input token IDs. Shape: (batch_size, seq_len).
                                               Values should be integers in range [0, vocab_size).
            attention_mask (Optional[torch.Tensor]): Mask to avoid performing attention on
                                                    padding tokens. Values should be 0 for tokens
                                                    to mask and 1 for tokens to attend to.
                                                    Shape: (batch_size, seq_len). Defaults to None.
            past_key_values (Optional[List[Tuple[torch.Tensor, torch.Tensor]]]): Cached key-value
                                                                                pairs from previous
                                                                                decoding steps. Used
                                                                                for efficient
                                                                                autoregressive
                                                                                generation. Each tuple
                                                                                contains (key, value)
                                                                                tensors. Defaults to None.
            use_cache (bool): Whether to return key-value caches for efficient generation.
                             When True, the model returns present key-value pairs that can be
                             used in subsequent forward passes. Defaults to False.
            **kwargs: Additional keyword arguments (unused, kept for compatibility).

        Returns:
            Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]: A tuple containing:
                - hidden_states (torch.Tensor): Final hidden states after processing through all layers.
                                               Shape: (batch_size, seq_len, hidden_size).
                - presents (List[Tuple[torch.Tensor, torch.Tensor]]): Updated key-value caches for
                                                                      each layer if use_cache is True.
                                                                      Each tuple contains (key, value)
                                                                      tensors for one layer.
                - aux_loss (torch.Tensor): Auxiliary loss from MoE layers. This is the sum of
                                          load-balancing losses from all MoE layers. Scalar tensor.
                                          Will be 0.0 if no MoE layers are used.
        """
        batch_size, seq_length = input_ids.shape
        past_key_values = past_key_values or [None] * len(self.layers)
        
        # Determine starting position for RoPE (important for cached generation)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # Convert token IDs to embeddings and apply dropout
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # Extract position embeddings for the current sequence
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        # Process through all Transformer layers
        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        # Apply final layer normalization
        hidden_states = self.norm(hidden_states)

        # Compute auxiliary loss from MoE layers (if any)
        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )

        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """
    MiniMind model with a language modeling head for causal language modeling tasks.

    This class extends the base MiniMind model with a linear classification layer for next-token
    prediction, making it suitable for causal language modeling tasks such as text generation,
    completion, and chat applications. It inherits from HuggingFace's PreTrainedModel and
    GenerationMixin to provide compatibility with the transformers ecosystem and advanced
    generation capabilities.

    The model architecture consists of:
    1. **Base Model**: The core MiniMindModel that processes input tokens
    2. **Language Modeling Head**: A linear layer that projects hidden states to vocabulary logits
    3. **Weight Sharing**: The embedding weights are tied with the LM head weights for efficiency

    Key Features:
    - Causal (autoregressive) language modeling
    - Support for key-value caching during generation
    - Compatible with HuggingFace generation utilities
    - Mixture-of-Experts (MoE) support with auxiliary loss
    - Efficient memory usage through weight tying

    Attributes:
        config_class (type): The configuration class for this model (MiniMindConfig).
        config (MiniMindConfig): Configuration object containing model hyperparameters.
        model (MiniMindModel): The core Transformer model.
        lm_head (nn.Linear): Linear layer for language modeling, mapping hidden states to vocabulary logits.
                            Shape: (hidden_size, vocab_size). Weights are tied with embed_tokens.
        OUT (CausalLMOutputWithPast): Output container for structured return values.
    """
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        """
        Initialize the MiniMind causal language model.

        Args:
            config (MiniMindConfig, optional): Configuration object containing model hyperparameters.
                                              If None, uses default MiniMindConfig. Defaults to None.
        """
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        
        # Initialize the core Transformer model
        self.model = MiniMindModel(self.config)
        
        # Language modeling head - projects hidden states to vocabulary probabilities
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        
        # Weight tying: share parameters between token embeddings and LM head
        # This reduces the number of parameters and often improves performance
        self.model.embed_tokens.weight = self.lm_head.weight
        
        # Output container for structured return values
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        """
        Forward pass for causal language modeling.

        This method processes input tokens through the model and computes next-token prediction
        logits. It supports efficient generation through key-value caching and provides options
        to reduce memory usage by only computing logits for a subset of positions.

        Args:
            input_ids (Optional[torch.Tensor]): Input token IDs. Shape: (batch_size, seq_len).
                                               Values should be integers in range [0, vocab_size).
                                               Required for forward pass.
            attention_mask (Optional[torch.Tensor]): Mask to avoid performing attention on
                                                    padding tokens. Values should be 0 for tokens
                                                    to mask and 1 for tokens to attend to.
                                                    Shape: (batch_size, seq_len). Defaults to None.
            past_key_values (Optional[List[Tuple[torch.Tensor, torch.Tensor]]]): Cached key-value
                                                                                pairs from previous
                                                                                decoding steps. Each
                                                                                tuple contains (key, value)
                                                                                for one layer. Used for
                                                                                efficient autoregressive
                                                                                generation. Defaults to None.
            use_cache (bool): Whether to return key-value caches for efficient generation.
                             When True, enables faster subsequent forward passes by reusing
                             computed attention states. Defaults to False.
            logits_to_keep (Union[int, torch.Tensor]): Controls which positions to compute logits for:
                                                      - If int: keeps the last N positions (0 means all)
                                                      - If tensor: boolean mask or indices specifying positions
                                                      This optimization reduces memory usage during generation.
                                                      Defaults to 0 (keep all).
            **args: Additional keyword arguments passed to the base model.

        Returns:
            CausalLMOutputWithPast: A structured output containing:
                - logits (torch.Tensor): Next-token prediction logits for the specified positions.
                                        Shape: (batch_size, logits_positions, vocab_size).
                                        Use torch.softmax(logits, dim=-1) to get probabilities.
                - last_hidden_state (torch.Tensor): Hidden states from the final layer.
                                                   Shape: (batch_size, seq_len, hidden_size).
                - past_key_values (List[Tuple[torch.Tensor, torch.Tensor]]): Updated key-value caches
                                                                            if use_cache is True.
                - aux_loss (torch.Tensor): Auxiliary loss from MoE layers for load balancing.
                                          Scalar tensor. Add to main loss during training.

        Note:
            During training, typically logits_to_keep=0 to compute loss over all positions.
            During generation, logits_to_keep=1 to only compute logits for the last position,
            significantly reducing memory usage and computation.
        """
        # Process input through the base Transformer model
        h, past_kvs, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        
        # Determine which positions to compute logits for (memory optimization)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        
        # Compute next-token prediction logits for selected positions
        logits = self.lm_head(h[:, slice_indices, :])
        
        # Populate structured output container
        self.OUT.__setitem__('last_hidden_state', h)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        
        return self.OUT
