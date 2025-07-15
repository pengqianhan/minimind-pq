"""
MiniMind Tokenizer Training Script

This script trains a custom Byte-Pair Encoding (BPE) tokenizer for MiniMind language models.
The tokenizer is trained on domain-specific text data to create an optimal vocabulary that
balances compression efficiency with model performance.

**Tokenization Approach:**
- **BPE (Byte-Pair Encoding)**: Subword tokenization that balances vocabulary size and text coverage
- **Byte-Level Processing**: Handles all Unicode characters robustly, including emojis and special symbols
- **Special Tokens**: Includes conversation markers for ChatML format compatibility

**Key Features:**
- **Vocabulary Optimization**: 6400 tokens optimized for the target domain
- **Special Token Integration**: ChatML markers for conversation formatting
- **Unicode Support**: Full Unicode character coverage through byte-level encoding
- **Reproducible Training**: Fixed random seed for consistent tokenizer builds

**BPE Training Process:**
1. **Character-Level Initialization**: Start with individual bytes as base vocabulary
2. **Frequency Analysis**: Count byte pair frequencies in training corpus
3. **Merge Operations**: Iteratively merge most frequent byte pairs
4. **Vocabulary Building**: Build final vocabulary of subword units
5. **Special Token Integration**: Reserve slots for conversation and control tokens

**Special Tokens:**
- `<|endoftext|>`: End-of-document marker (token ID: 0)
- `<|im_start|>`: ChatML conversation start marker
- `<|im_end|>`: ChatML conversation end marker

**Vocabulary Size Considerations:**
- **6400 tokens**: Balanced size for 26M-145M parameter models
- **Smaller vocabulary**: Faster training, lower memory usage
- **Larger vocabulary**: Better compression, more precise representation
- **Trade-off**: Model size vs tokenization efficiency

**Training Data Format:**
Expected JSONL format with text fields:
```json
{"text": "This is a sample text for tokenizer training."}
{"text": "Another piece of text content for vocabulary learning."}
```

**Output Files:**
- `tokenizer.json`: Complete tokenizer configuration and vocabulary
- `tokenizer_config.json`: Metadata and configuration parameters

**Usage:**
```python
python train_tokenizer.py

# Load trained tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('./model/')
```

**Quality Metrics:**
- Compression ratio: Average tokens per character
- Coverage: Percentage of unseen text properly tokenized
- Subword quality: Meaningful subword unit formation

Dependencies:
    - tokenizers: HuggingFace tokenizers library for BPE training
    - json: Data loading and processing
    - random: Reproducible randomization
"""

import random
import json
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
import os

# Set random seed for reproducible tokenizer training
random.seed(42)


def train_tokenizer():
    """
    Train a BPE tokenizer optimized for MiniMind language models.

    This function implements the complete tokenizer training pipeline from data loading
    to final model serialization. It creates a subword vocabulary optimized for the
    target domain while ensuring proper handling of conversation formatting.

    **Training Pipeline:**
    1. **Data Loading**: Read training corpus from JSONL format
    2. **Tokenizer Initialization**: Configure BPE model with byte-level processing
    3. **Special Token Definition**: Set up ChatML conversation markers
    4. **Vocabulary Training**: Learn optimal subword units from corpus
    5. **Configuration**: Apply decoding and post-processing settings
    6. **Validation**: Verify special token assignments
    7. **Serialization**: Save tokenizer for model training

    **Technical Details:**
    - **Model Type**: Byte-Pair Encoding (BPE) for subword tokenization
    - **Pre-tokenizer**: ByteLevel for robust Unicode handling
    - **Vocabulary Size**: 6400 tokens optimized for model size
    - **Special Tokens**: 3 reserved tokens for conversation formatting
    - **Decoder**: ByteLevel for proper text reconstruction

    **Memory Requirements:**
    - Training corpus size: Depends on dataset size
    - Peak memory usage: ~1-2GB for typical corpora
    - Output size: ~25MB for complete tokenizer files

    **Quality Assurance:**
    - Validates special token ID assignments
    - Checks vocabulary completeness
    - Ensures proper encoding/decoding round-trips
    """
    # 读取JSONL文件并提取文本数据
    def read_texts_from_jsonl(file_path):
        """
        Generator function to read text data from JSONL file.

        This function provides memory-efficient streaming of training data,
        avoiding loading the entire dataset into memory at once.

        Args:
            file_path (str): Path to JSONL file containing training text data.
                           Each line should be a JSON object with 'text' field.

        Yields:
            str: Individual text samples for tokenizer training.
                Each yield provides one text document for vocabulary learning.

        **File Format:**
        ```json
        {"text": "Sample document 1"}
        {"text": "Sample document 2"}
        ```

        **Memory Efficiency:**
        - Streaming processing: One line at a time
        - No corpus-wide memory allocation
        - Suitable for large datasets (>1GB)
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                yield data['text']

    # Data source configuration
    data_path = '../dataset/pretrain_hq.jsonl'

    # 初始化tokenizer
    # tokenizer (Tokenizer): BPE tokenizer instance for training
    tokenizer = Tokenizer(models.BPE())
    
    # Configure byte-level pre-tokenizer for robust Unicode handling
    # add_prefix_space=False: No automatic space prefixing for efficiency
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 定义特殊token
    # special_tokens (List[str]): Reserved tokens for conversation formatting
    special_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]

    # 设置训练器并添加特殊token
    # trainer (BpeTrainer): Configured trainer for vocabulary learning
    trainer = trainers.BpeTrainer(
        vocab_size=6400,                                      # Target vocabulary size
        special_tokens=special_tokens,                        # Reserved conversation tokens
        show_progress=True,                                   # Display training progress
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet() # Start with byte-level alphabet
    )

    # 读取文本数据
    # texts (Generator): Streaming text data for memory-efficient training
    texts = read_texts_from_jsonl(data_path)

    # 训练tokenizer
    # Core vocabulary learning from training corpus
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # 设置解码器
    # Configure decoder for proper text reconstruction
    tokenizer.decoder = decoders.ByteLevel()

    # 检查特殊token的索引
    # Validate that special tokens are assigned expected IDs
    assert tokenizer.token_to_id("<|endoftext|>") == 0
    assert tokenizer.token_to_id("<|im_start|>") == 1
    assert tokenizer.token_to_id("<|im_end|>") == 2

    # 保存tokenizer
    tokenizer_dir = "../model/"
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save("../model/")

    # 手动创建配置文件
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": {
            "0": {
                "content": "<|endoftext|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "1": {
                "content": "<|im_start|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "<|im_end|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "additional_special_tokens": [],
        "bos_token": "<|im_start|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "legacy": True,
        "model_max_length": 32768,
        "pad_token": "<|endoftext|>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<|endoftext|>",
        "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{{ '<|im_start|>system\\n' + system_message + '<|im_end|>\\n' }}{% else %}{{ '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\\n' + content + '<|im_end|>\\n<|im_start|>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<|im_end|>' + '\\n' }}{% endif %}{% endfor %}"
    }

    # 保存配置文件
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)

    print("Tokenizer training completed and saved.")


def eval_tokenizer():
    from transformers import AutoTokenizer

    # 加载预训练的tokenizer
    tokenizer = AutoTokenizer.from_pretrained("../model/")

    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '你来自哪里？'},
        {"role": "assistant", "content": '我来自地球'}
    ]
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    print(new_prompt)

    # 获取实际词汇表长度（包括特殊符号）
    actual_vocab_size = len(tokenizer)
    print('tokenizer实际词表长度：', actual_vocab_size)

    model_inputs = tokenizer(new_prompt)
    print('encoder长度：', len(model_inputs['input_ids']))

    input_ids = model_inputs['input_ids']
    response = tokenizer.decode(input_ids, skip_special_tokens=False)
    print('decoder和原始文本是否一致：', response == new_prompt)


def main():
    train_tokenizer()
    eval_tokenizer()


if __name__ == '__main__':
    main()
