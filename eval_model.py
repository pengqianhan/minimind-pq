"""
MiniMind Model Evaluation and Interactive Chat Interface

This module provides a comprehensive evaluation and interactive chat interface for MiniMind language models.
It supports multiple model configurations, loading strategies, and generation parameters to facilitate
both automated testing and interactive conversations with the model.

Key Features:
- **Multiple Model Loading**: Support for both native PyTorch checkpoints and HuggingFace transformers format
- **LoRA Integration**: Apply Low-Rank Adaptation weights for domain-specific fine-tuning
- **Flexible Evaluation**: Automated testing with predefined prompts or interactive manual input
- **Conversation History**: Maintain context across multiple conversation turns
- **Streaming Output**: Real-time token-by-token response generation
- **Reproducible Generation**: Configurable random seeds for consistent or varied outputs

Model Architectures Supported:
- MiniMind2-Small (26M parameters): hidden_size=512, num_hidden_layers=8
- MiniMind2 (104M parameters): hidden_size=768, num_hidden_layers=16  
- MiniMind2-MoE (145M parameters): hidden_size=640, num_hidden_layers=8, use_moe=True

Model Training Stages:
- Stage 0 (Pretrain): Base language modeling for text completion
- Stage 1 (SFT): Supervised fine-tuning for conversational abilities
- Stage 2 (RLHF): Reinforcement learning from human feedback
- Stage 3 (Reason): Enhanced reasoning capabilities
- Stage 4 (GRPO): Additional preference optimization

Usage Examples:
    # Interactive chat with default model
    python eval_model.py
    
    # Automated evaluation with medical LoRA
    python eval_model.py --lora_name lora_medical --model_mode 1
    
    # Load specific model configuration
    python eval_model.py --hidden_size 768 --num_hidden_layers 16 --temperature 0.7

Dependencies:
    - torch: Core deep learning framework
    - transformers: HuggingFace model loading and tokenization
    - numpy: Numerical computations
    - argparse: Command-line argument parsing
    - random: Random number generation for reproducibility
"""

import argparse
import random
import warnings
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *

warnings.filterwarnings('ignore')


def init_model(args):
    """
    Initializes and loads the language model and tokenizer.

    This function handles two main model loading scenarios:
    1. Loading a native PyTorch model checkpoint (`.pth` file). This includes options for
       different model configurations like Mixture-of-Experts (MoE) and applying LoRA weights.
    2. Loading a model from the Hugging Face Hub (transformers format).

    Args:
        args (argparse.Namespace): An object containing command-line arguments. Key arguments include:
            - load (int): 0 for native torch weights, 1 for transformers model.
            - out_dir (str): Directory where native model checkpoints are stored.
            - model_mode (int): The mode of the model to load (e.g., pretrain, sft, rlhf).
            - hidden_size (int): The hidden size of the model.
            - num_hidden_layers (int): The number of hidden layers in the model.
            - use_moe (bool): Whether the model uses a Mixture-of-Experts architecture.
            - device (str): The device to load the model on ('cuda' or 'cpu').
            - lora_name (str): The name of the LoRA weights to apply. If 'None', no LoRA is applied.

    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The loaded and evaluated language model, moved to the specified device.
            - tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
    """
    tokenizer = AutoTokenizer.from_pretrained('./model/')
    if args.load == 0:
        moe_path = '_moe' if args.use_moe else ''
        modes = {0: 'pretrain', 1: 'full_sft', 2: 'rlhf', 3: 'reason', 4: 'grpo'}
        ckp = f'./{args.out_dir}/{modes[args.model_mode]}_{args.hidden_size}{moe_path}.pth'

        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=args.use_moe
        ))

        # Load the state dictionary from the checkpoint file.
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)

        # If a LoRA name is provided, apply LoRA modifications and load the weights.
        if args.lora_name != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.out_dir}/lora/{args.lora_name}_{args.hidden_size}.pth')
    else:
        # Load a model pre-packaged in the Hugging Face transformers format.
        transformers_model_path = './MiniMind2'
        tokenizer = AutoTokenizer.from_pretrained(transformers_model_path)
        model = AutoModelForCausalLM.from_pretrained(transformers_model_path, trust_remote_code=True)
    # Print the total number of trainable parameters in the model.
    print(f'MiniMind模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M(illion)')
    return model.eval().to(args.device), tokenizer


def get_prompt_datas(args):
    """
    Retrieves a list of prompt strings based on the model's mode and LoRA configuration.

    This function provides different sets of prompts for various evaluation scenarios:
    - Pre-trained models: Prompts are designed to test continuation/completion abilities.
    - Fine-tuned models (without LoRA): Generic, open-ended questions for dialogue evaluation.
    - LoRA-specific models: Prompts are tailored to the specific domain of the LoRA weights
      (e.g., identity-related questions, medical inquiries).

    Args:
        args (argparse.Namespace): An object containing command-line arguments. Key arguments are:
            - model_mode (int): The mode of the model. 0 for pre-trained, other values for fine-tuned.
            - lora_name (str): The name of the LoRA weights being used. Used to select
              domain-specific prompts. 'None' implies a general-purpose model.

    Returns:
        list[str]: A list of strings, where each string is a prompt for the model.
    """
    if args.model_mode == 0:
        # pretrain模型的接龙能力（无法对话）
        prompt_datas = [
            '马克思主义基本原理',
            '人类大脑的主要功能',
            '万有引力原理是',
            '世界上最高的山峰是',
            '二氧化碳在空气中',
            '地球上最大的动物有',
            '杭州市的美食有'
        ]
    else:
        if args.lora_name == 'None':
            # 通用对话问题
            prompt_datas = [
                '请介绍一下自己。',
                '你更擅长哪一个学科？',
                '鲁迅的《狂人日记》是如何批判封建礼教的？',
                '我咳嗽已经持续了两周，需要去医院检查吗？',
                '详细的介绍光速的物理概念。',
                '推荐一些杭州的特色美食吧。',
                '请为我讲解“大语言模型”这个概念。',
                '如何理解ChatGPT？',
                'Introduce the history of the United States, please.'
            ]
        else:
            # 特定领域问题
            lora_prompt_datas = {
                'lora_identity': [
                    "你是ChatGPT吧。",
                    "你叫什么名字？",
                    "你和openai是什么关系？"
                ],
                'lora_medical': [
                    '我最近经常感到头晕，可能是什么原因？',
                    '我咳嗽已经持续了两周，需要去医院检查吗？',
                    '服用抗生素时需要注意哪些事项？',
                    '体检报告中显示胆固醇偏高，我该怎么办？',
                    '孕妇在饮食上需要注意什么？',
                    '老年人如何预防骨质疏松？',
                    '我最近总是感到焦虑，应该怎么缓解？',
                    '如果有人突然晕倒，应该如何急救？'
                ],
            }
            prompt_datas = lora_prompt_datas[args.lora_name]

    return prompt_datas


# 设置可复现的随机种子
def setup_seed(seed):
    """
    Sets the random seed for all relevant libraries to ensure reproducibility.

    This function sets the seed for Python's `random`, `numpy`, and `torch` (for both CPU and CUDA).
    It also configures CUDA operations to be deterministic.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """
    Main function to run interactive chat or automated evaluation with the MiniMind model.

    This function orchestrates the complete evaluation pipeline from model initialization to
    response generation. It provides two operational modes:
    
    **Automated Testing Mode (test_mode=0):**
    - Iterates through predefined prompts based on model configuration
    - Generates responses automatically for systematic evaluation
    - Useful for benchmarking and consistent testing across model versions
    
    **Interactive Chat Mode (test_mode=1):**
    - Accepts user input in real-time for conversational interaction
    - Maintains conversation history for context-aware responses
    - Provides streaming output for better user experience

    **Process Flow:**
    1. **Argument Parsing**: Configure model architecture, generation parameters, and evaluation settings
    2. **Model Initialization**: Load model weights and tokenizer (supports both PyTorch and HuggingFace formats)
    3. **Prompt Selection**: Choose appropriate prompt set based on model mode and LoRA configuration
    4. **Generation Loop**: Process prompts and generate responses with optional conversation history
    5. **Output Streaming**: Display responses token-by-token with proper formatting

    **Variable Documentation:**
    - args (argparse.Namespace): Configuration object containing all command-line arguments
    - model (torch.nn.Module): Loaded MiniMind model instance, moved to specified device
    - tokenizer (PreTrainedTokenizer): Tokenizer for text encoding/decoding operations
    - prompts (List[str]): List of prompt strings for evaluation or interaction
    - test_mode (int): Operational mode - 0 for automated testing, 1 for interactive input
    - streamer (TextStreamer): HuggingFace TextStreamer for real-time token output
    - messages (List[Dict[str, str]]): Conversation history buffer with role-content pairs
    - new_prompt (str): Formatted prompt string with chat template and history context
    - inputs (Dict[str, torch.Tensor]): Tokenized input tensors ready for model consumption
    - generated_ids (torch.Tensor): Model output token IDs including input and generated tokens
    - response (str): Decoded response text excluding special tokens and input prompt

    **Configuration Parameters:**
    - Model Architecture: hidden_size, num_hidden_layers, use_moe determine model capacity
    - Generation Settings: temperature (randomness), top_p (nucleus sampling), max_seq_len (length limit)
    - Context Management: history_cnt controls conversation memory span
    - Loading Strategy: load parameter selects between PyTorch (.pth) and HuggingFace formats
    - Domain Specialization: lora_name applies domain-specific adaptations (medical, identity, etc.)

    **Error Handling:**
    - Graceful handling of CUDA availability (falls back to CPU)
    - Input validation for numeric parameters
    - Model loading verification with informative error messages

    **Performance Considerations:**
    - Key-value caching enabled for efficient autoregressive generation
    - Truncation applied to prevent memory overflow with long conversations
    - Optional deterministic seeding for reproducible evaluation
    """
    parser = argparse.ArgumentParser(description="Chat with MiniMind")

    # --- General Configuration ---
    parser.add_argument('--lora_name', default='None', type=str,
                        help="Name of the LoRA weights to apply (e.g., 'lora_medical'). 'None' for no LoRA.")
    parser.add_argument('--out_dir', default='out', type=str,
                        help="Directory where model checkpoints are stored.")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str,
                        help="Device to run the model on ('cuda' or 'cpu').")

    # --- Generation Parameters ---
    parser.add_argument('--temperature', default=0.85, type=float,
                        help="Controls randomness in generation. Lower is more deterministic. (float): range [0.0, 1.0]")
    parser.add_argument('--top_p', default=0.85, type=float,
                        help="Nucleus sampling threshold. (float): range [0.0, 1.0]")
    parser.add_argument('--max_seq_len', default=8192, type=int,
                        help="Maximum length of the generated sequence. Note: This does not imply the model can handle "
                             "context of this length effectively, but prevents truncation.")

    # --- Model Architecture ---
    # MiniMind2-moe (145M): (hidden_size=640, num_hidden_layers=8, use_moe=True)
    # MiniMind2-Small (26M): (hidden_size=512, num_hidden_layers=8)
    # MiniMind2 (104M): (hidden_size=768, num_hidden_layers=16)
    parser.add_argument('--hidden_size', default=512, type=int,
                        help="The hidden size (embedding dimension) of the model. (int)")
    parser.add_argument('--num_hidden_layers', default=8, type=int,
                        help="The number of transformer layers in the model. (int)")
    parser.add_argument('--use_moe', default=False, type=bool,
                        help="Whether the model uses a Mixture-of-Experts architecture. (bool)")

    # --- History and Loading ---
    parser.add_argument('--history_cnt', default=0, type=int,
                        help="Number of previous conversation turns to include as context. Must be an even number "
                             "(1 user + 1 assistant = 1 turn). 0 to disable history.")
    parser.add_argument('--load', default=0, type=int,
                        help="Model loading mode. 0: native PyTorch weights (.pth), 1: Hugging Face transformers format.")
    parser.add_argument('--model_mode', default=1, type=int,
                        help="Specifies the model's training stage. 0: Pre-trained, 1: SFT-Chat, 2: RLHF-Chat, "
                             "3: Reason, 4: RLAIF-Chat.")

    args = parser.parse_args()

    # Initialize model and tokenizer based on configuration
    model, tokenizer = init_model(args)

    # Retrieve appropriate prompt set for evaluation
    prompts = get_prompt_datas(args)
    
    # User selects between automated testing and interactive mode
    test_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    
    # Initialize streaming output handler for real-time token display
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # messages (List[Dict[str, str]]): Conversation history buffer storing role-content pairs
    # Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    messages = []
    
    # Main generation loop: iterate through prompts or accept user input
    for idx, prompt in enumerate(prompts if test_mode == 0 else iter(lambda: input('👶: '), '')):
        # Set random seed for generation variability or reproducibility
        # Each iteration uses a different seed for diverse outputs
        setup_seed(random.randint(0, 2048))
        # setup_seed(2025)  # Uncomment for fixed seed and reproducible outputs

        if test_mode == 0: print(f'👶: {prompt}')

        # Maintain conversation context within specified history window
        # Trim to last N turns to prevent context overflow
        messages = messages[-args.history_cnt:] if args.history_cnt else []
        messages.append({"role": "user", "content": prompt})

        # Format conversation with chat template for proper model input structure
        # Pre-trained models use simple BOS token, fine-tuned models use structured templates
        new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ) if args.model_mode != 0 else (tokenizer.bos_token + prompt)

        # inputs (Dict[str, torch.Tensor]): Tokenized input ready for model consumption
        # Contains 'input_ids' (token IDs) and 'attention_mask' (padding indicators)
        inputs = tokenizer(
            new_prompt,
            return_tensors="pt",
            truncation=True
        ).to(args.device)

        print('🤖️: ', end='')
        
        # Generate model response with specified parameters
        # generated_ids (torch.Tensor): Complete token sequence [input_tokens + generated_tokens]
        # Shape: (batch_size=1, total_sequence_length)
        generated_ids = model.generate(
            inputs["input_ids"],                    # Input token IDs (torch.Tensor)
            max_new_tokens=args.max_seq_len,       # Maximum new tokens to generate (int)
            num_return_sequences=1,                # Number of response variants (int)
            do_sample=True,                        # Enable sampling-based generation (bool)
            attention_mask=inputs["attention_mask"], # Padding mask (torch.Tensor)
            pad_token_id=tokenizer.pad_token_id,   # Padding token ID (int)
            eos_token_id=tokenizer.eos_token_id,   # End-of-sequence token ID (int)
            streamer=streamer,                     # Real-time output streamer (TextStreamer)
            top_p=args.top_p,                      # Nucleus sampling threshold (float)
            temperature=args.temperature           # Sampling temperature for randomness (float)
        )

        # Extract and decode only the newly generated tokens (excluding input prompt)
        # response (str): Clean response text with special tokens removed
        response = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # Append model response to conversation history for context in next turn
        messages.append({"role": "assistant", "content": response})
        print('\n\n')


if __name__ == "__main__":
    main()
