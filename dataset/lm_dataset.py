"""
MiniMind Dataset Classes for Language Model Training

This module provides PyTorch Dataset implementations for different stages of language model training.
Each dataset class handles specific data formats and training requirements for various training paradigms
used in the MiniMind model pipeline.

Dataset Classes Overview:
1. **PretrainDataset**: For unsupervised language modeling pre-training
2. **SFTDataset**: For supervised fine-tuning with conversational data
3. **DPODataset**: For Direct Preference Optimization training with preference pairs
4. **RLAIFDataset**: For Reinforcement Learning from AI Feedback training

Key Features:
- **Tokenization Management**: Consistent tokenization across all dataset types
- **Loss Masking**: Intelligent masking to compute loss only on relevant tokens
- **Memory Efficiency**: Configurable sequence lengths and padding strategies
- **Chat Formatting**: Support for ChatML conversation templates
- **Dynamic Loading**: Efficient data loading from JSONL files

Data Flow Pipeline:
1. Raw text data → Tokenization → Tensor conversion → Loss mask generation
2. Sequence truncation/padding for batch consistency
3. Input-output pair creation for autoregressive training

Technical Specifications:
- Input Format: JSONL files with structured conversation data
- Tokenization: HuggingFace tokenizer compatibility
- Output Format: PyTorch tensors ready for model training
- Memory Management: Configurable sequence lengths and batch sizes

Training Stage Applications:
- **Pretrain**: Learn basic language patterns from raw text
- **SFT**: Learn conversational abilities from instruction-response pairs  
- **DPO**: Learn preferences between response alternatives
- **RLAIF**: Learn from AI-generated feedback and rankings

Dependencies:
    - torch: PyTorch framework for tensor operations
    - transformers: HuggingFace tokenizers and utilities
    - json: JSON data parsing
    - pandas: Data manipulation (if needed)
    - numpy: Numerical operations
    - sklearn: Data splitting utilities
"""

import json
import random
import re

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import os
import ast

# Disable tokenizer parallelism to avoid warnings in multi-processing environments
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    """
    Dataset class for unsupervised language model pre-training.

    This dataset processes raw text data for next-token prediction training, which is the foundation
    of language model pre-training. It handles tokenization, sequence preparation, and loss masking
    for efficient causal language modeling.

    **Training Objective:**
    Learn to predict the next token given previous context, enabling the model to capture
    fundamental language patterns, syntax, grammar, and basic world knowledge.

    **Data Format:**
    Expects JSONL files where each line contains a JSON object with a 'text' field:
    ```json
    {"text": "This is a sample text for language modeling."}
    {"text": "Another piece of text for training the model."}
    ```

    **Processing Pipeline:**
    1. Load raw text from JSONL file
    2. Tokenize text using provided tokenizer
    3. Create input-output pairs for autoregressive training (X[t] → Y[t+1])
    4. Apply padding to maintain consistent sequence lengths
    5. Generate loss masks to exclude padding tokens from loss computation

    **Memory Efficiency:**
    - Configurable maximum sequence length to control memory usage
    - Automatic padding and truncation for batch consistency
    - Lazy loading of samples to minimize RAM usage

    Attributes:
        tokenizer (PreTrainedTokenizer): HuggingFace tokenizer for text encoding/decoding.
        max_length (int): Maximum sequence length for input processing. Longer sequences
                         are truncated, shorter ones are padded.
        samples (List[Dict]): List of loaded data samples from the JSONL file.
                             Each sample contains a 'text' field with raw text content.
    """
    def __init__(self, data_path, tokenizer, max_length=512):
        """
        Initialize the PretrainDataset.

        Args:
            data_path (str): Path to the JSONL file containing raw text data.
                           Each line should be a JSON object with a 'text' field.
            tokenizer (PreTrainedTokenizer): HuggingFace tokenizer instance for text processing.
                                           Must have pad_token_id defined for proper masking.
            max_length (int, optional): Maximum sequence length for tokenization and processing.
                                      Sequences longer than this will be truncated, shorter ones
                                      will be padded. Defaults to 512.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        """
        Load and parse data from a JSONL file.

        This method reads the input file line by line, parsing each line as a JSON object.
        It extracts the text content and stores it for later processing.

        Args:
            path (str): Path to the JSONL file to load.

        Returns:
            List[Dict]: List of parsed JSON objects, each containing text data.
                       Format: [{"text": "sample text 1"}, {"text": "sample text 2"}, ...]

        Raises:
            FileNotFoundError: If the specified file path does not exist.
            json.JSONDecodeError: If any line in the file is not valid JSON.
        """
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of text samples available for training.
        """
        return len(self.samples)

    def __getitem__(self, index):
        """
        Retrieve and process a single training sample.

        This method converts raw text into tensors suitable for causal language modeling.
        It creates input-output pairs where the model learns to predict token t+1 given tokens 0 to t.

        **Processing Steps:**
        1. Extract text from sample and tokenize it
        2. Apply truncation and padding to reach max_length
        3. Create input sequence X (tokens 0 to n-1) and target sequence Y (tokens 1 to n)
        4. Generate loss mask to exclude padding tokens from loss computation

        Args:
            index (int): Index of the sample to retrieve (0-based).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - X (torch.Tensor): Input token IDs for the model. Shape: (max_length-1,).
                                   Contains token IDs from positions 0 to max_length-2.
                - Y (torch.Tensor): Target token IDs for next-token prediction. Shape: (max_length-1,).
                                   Contains token IDs from positions 1 to max_length-1.
                - loss_mask (torch.Tensor): Binary mask indicating which positions to include
                                           in loss computation. Shape: (max_length-1,).
                                           1 for real tokens, 0 for padding tokens.

        **Tensor Shapes:**
        - Original tokenized sequence: (max_length,) 
        - X (input): (max_length-1,) - all tokens except the last
        - Y (target): (max_length-1,) - all tokens except the first  
        - loss_mask: (max_length-1,) - excludes padding from loss

        **Example:**
        ```
        Original text: "Hello world"
        Tokenized: [BOS, 15496, 1917, PAD, PAD]  # length=5
        X: [BOS, 15496, 1917, PAD]              # length=4, predict next token
        Y: [15496, 1917, PAD, PAD]              # length=4, targets for X
        loss_mask: [1, 1, 0, 0]                 # length=4, ignore PAD in loss
        ```
        """
        sample = self.samples[index]

        # Tokenize the text with truncation and padding
        # encoding (transformers.BatchEncoding): Contains input_ids, attention_mask, etc.
        encoding = self.tokenizer(
            str(sample['text']),                # Raw text content (str)
            max_length=self.max_length,         # Maximum sequence length (int)
            padding='max_length',               # Pad to max_length (str)
            truncation=True,                    # Truncate if longer (bool)
            return_tensors='pt'                 # Return PyTorch tensors (str)
        )
        
        # Extract token IDs and remove batch dimension
        # input_ids (torch.Tensor): Token IDs with shape (max_length,)
        input_ids = encoding.input_ids.squeeze()
        
        # Create loss mask: 1 for real tokens, 0 for padding tokens
        # loss_mask (torch.Tensor): Binary mask with shape (max_length,)
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        # Create input-output pairs for autoregressive training
        # Shift by one position: model sees X[0:n-1] and predicts Y[1:n]
        X = torch.tensor(input_ids[:-1], dtype=torch.long)      # Input tokens (max_length-1,)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)       # Target tokens (max_length-1,)  
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # Aligned loss mask (max_length-1,)

        return X, Y, loss_mask


class SFTDataset(Dataset):
    """
    Dataset class for Supervised Fine-Tuning (SFT) with conversational data.

    This dataset processes multi-turn conversation data to fine-tune pre-trained language models
    for instruction-following and conversational capabilities. It implements intelligent loss masking
    to ensure the model only learns from assistant responses, not user inputs.

    **Training Objective:**
    Learn to generate appropriate responses in conversational contexts by training only on
    assistant utterances while using the full conversation history as context.

    **Data Format:**
    Expects JSONL files with conversation structures:
    ```json
    {
      "conversations": [
        {"content": "What is machine learning?"},
        {"content": "Machine learning is a subset of artificial intelligence..."},
        {"content": "Can you give an example?"},
        {"content": "Sure! A common example is email spam detection..."}
      ]
    }
    ```

    **Loss Masking Strategy:**
    - **User turns**: Loss masked (model doesn't learn to predict user inputs)
    - **Assistant turns**: Loss unmasked (model learns to generate responses)
    - **Padding tokens**: Loss masked (excluded from training)

    **ChatML Template Format:**
    Conversations are formatted using ChatML structure:
    ```
    <|im_start|>user
    What is machine learning?<|im_end|>
    <|im_start|>assistant
    Machine learning is a subset...<|im_end|>
    ```

    **Key Features:**
    - Automatic role assignment (user/assistant based on turn position)
    - Dynamic loss mask generation for selective training
    - ChatML template formatting for conversation structure
    - Configurable sequence length management

    Attributes:
        tokenizer (PreTrainedTokenizer): HuggingFace tokenizer with ChatML template support.
        max_length (int): Maximum sequence length for conversation processing.
        samples (List[Dict]): Loaded conversation samples from JSONL file.
        bos_id (List[int]): Token IDs for assistant start marker '<|im_start|>assistant'.
        eos_id (List[int]): Token IDs for conversation end marker '<|im_end|>'.
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        """
        Initialize the SFTDataset.

        Args:
            jsonl_path (str): Path to JSONL file containing conversation data.
                            Each line should contain a JSON object with 'conversations' field.
            tokenizer (PreTrainedTokenizer): HuggingFace tokenizer with ChatML template support.
                                           Must have apply_chat_template method available.
            max_length (int, optional): Maximum sequence length for conversation processing.
                                      Longer conversations will be truncated. Defaults to 1024.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        
        # Extract token IDs for loss mask generation
        # bos_id (List[int]): Token sequence marking start of assistant response
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        # eos_id (List[int]): Token sequence marking end of conversation turn
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids

    def __len__(self):
        """
        Return the total number of conversation samples.

        Returns:
            int: Number of conversation samples available for training.
        """
        return len(self.samples)

    def load_data(self, path):
        """
        Load conversation data from JSONL file.

        Args:
            path (str): Path to the JSONL file containing conversation data.

        Returns:
            List[Dict]: List of conversation samples, each containing 'conversations' field
                       with alternating user/assistant turns.

        Raises:
            FileNotFoundError: If the specified file path does not exist.
            json.JSONDecodeError: If any line contains invalid JSON.
            KeyError: If conversation structure is missing required fields.
        """
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        """
        Convert conversation turns into ChatML formatted prompt.

        This method takes a list of conversation turns and formats them using the ChatML
        template structure, automatically assigning user/assistant roles based on turn position.

        Args:
            conversations (List[Dict]): List of conversation turns, each containing 'content' field.
                                      Format: [{"content": "user message"}, {"content": "assistant response"}]

        Returns:
            str: Formatted conversation string using ChatML template structure.
                Contains properly formatted user and assistant turns with special tokens.

        **Role Assignment Logic:**
        - Even-indexed turns (0, 2, 4, ...): Assigned 'user' role
        - Odd-indexed turns (1, 3, 5, ...): Assigned 'assistant' role

        **Example:**
        ```
        Input: [{"content": "Hello"}, {"content": "Hi there!"}]
        Output: "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there!<|im_end|>"
        ```
        """
        # messages (List[Dict[str, str]]): Structured conversation with explicit roles
        messages = []
        for i, turn in enumerate(conversations):
            # Alternate between user and assistant roles based on turn index
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
        
        # Apply ChatML template formatting
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,                    # Return string, not token IDs
            add_generation_prompt=False        # Don't add prompt for next response
        )

    def _generate_loss_mask(self, input_ids):
        """
        Generate loss mask to train only on assistant responses.

        This method creates a binary mask that excludes user inputs and padding tokens
        from loss computation, ensuring the model only learns from assistant responses.

        **Masking Logic:**
        1. Scan through token sequence to find assistant response regions
        2. Mark tokens between '<|im_start|>assistant' and '<|im_end|>' as trainable (1)
        3. Mark all other tokens (user inputs, special tokens, padding) as masked (0)
        4. Include the end token in the trainable region for proper sequence termination

        Args:
            input_ids (List[int]): Tokenized conversation as list of token IDs.

        Returns:
            List[int]: Binary loss mask with same length as input_ids.
                      1 indicates tokens to include in loss computation (assistant responses),
                      0 indicates tokens to exclude (user inputs, padding, special tokens).

        **Example:**
        ```
        Conversation: "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi<|im_end|>"
        Token IDs:    [1001, 1002, 15496, 1003, 1001, 1004, 12345, 1003]
        Loss Mask:    [0,    0,    0,     0,    0,    1,    1,     1   ]
        ```
        """
        # Initialize all positions as masked (0)
        loss_mask = [0] * len(input_ids)
        i = 0
        
        # Scan through token sequence to find assistant response regions
        while i < len(input_ids):
            # Check if current position starts an assistant response marker
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                # Found assistant start marker, find the content region
                start = i + len(self.bos_id)  # Position after '<|im_start|>assistant'
                end = start
                
                # Search for the corresponding end marker
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                
                # Mark the assistant response region as trainable
                # Include positions from after BOS to after EOS (inclusive)
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                
                # Move index past the processed region
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        
        return loss_mask

    def __getitem__(self, index):
        """
        Process and return a single conversation sample for training.

        This method converts a conversation into tensors suitable for supervised fine-tuning,
        with appropriate loss masking to focus training on assistant responses only.

        Args:
            index (int): Index of the conversation sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Training tensors containing:
                - X (torch.Tensor): Input token sequence. Shape: (max_length-1,).
                                   Contains tokens from positions 0 to max_length-2.
                - Y (torch.Tensor): Target token sequence. Shape: (max_length-1,).
                                   Contains tokens from positions 1 to max_length-1.
                - loss_mask (torch.Tensor): Binary loss mask. Shape: (max_length-1,).
                                           1 for assistant tokens to train on, 0 for masked tokens.

        **Processing Steps:**
        1. Extract conversation from sample
        2. Format conversation using ChatML template
        3. Tokenize and apply padding/truncation
        4. Generate loss mask for selective training
        5. Create input-output pairs for autoregressive training

        **Loss Mask Application:**
        - User inputs: Excluded from loss (mask = 0)
        - Assistant responses: Included in loss (mask = 1) 
        - Padding tokens: Excluded from loss (mask = 0)
        - Special tokens: Typically excluded (mask = 0)
        """
        sample = self.samples[index]
        
        # Convert conversation turns to ChatML formatted string
        # prompt (str): Formatted conversation with proper role markers
        prompt = self._create_chat_prompt(sample['conversations'])
        
        # Tokenize the conversation and apply length constraints
        # input_ids (List[int]): Token IDs truncated to max_length
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        
        # Apply padding to reach consistent sequence length
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # Generate loss mask for selective training on assistant responses
        # loss_mask (List[int]): Binary mask indicating trainable positions
        loss_mask = self._generate_loss_mask(input_ids)

        # Create autoregressive training pairs with aligned loss mask
        # X: Input sequence (all tokens except last)
        # Y: Target sequence (all tokens except first) 
        # loss_mask: Aligned to target positions for proper masking
        X = torch.tensor(input_ids[:-1], dtype=torch.long)          # Shape: (max_length-1,)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)           # Shape: (max_length-1,)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)   # Shape: (max_length-1,)

        return X, Y, loss_mask


class DPODataset(Dataset):
    """
    Dataset class for Direct Preference Optimization (DPO) training.

    This dataset processes preference pairs for training models to align with human preferences
    without requiring explicit reward modeling. It handles chosen vs rejected response pairs
    for the same prompt, enabling direct optimization of model preferences.

    **Training Objective:**
    Learn to prefer chosen responses over rejected ones for the same prompt through direct
    preference optimization, improving response quality and alignment with human values.

    **Data Format:**
    Expects JSONL files with preference pair structures:
    ```json
    {
      "chosen": [
        {"role": "user", "content": "Explain quantum computing"},
        {"role": "assistant", "content": "Quantum computing is a computing paradigm..."}
      ],
      "rejected": [
        {"role": "user", "content": "Explain quantum computing"},
        {"role": "assistant", "content": "Quantum computing is just faster computers..."}
      ]
    }
    ```

    **DPO Algorithm:**
    DPO trains the model to maximize the log probability of chosen responses while minimizing
    the log probability of rejected responses, using a reference model for stability.

    **Loss Computation:**
    - **Chosen responses**: Used to compute positive log-likelihood
    - **Rejected responses**: Used to compute negative log-likelihood  
    - **Loss masking**: Applied to focus training on assistant responses only

    **Key Features:**
    - Paired response processing for preference learning
    - Independent loss mask generation for chosen/rejected pairs
    - ChatML template formatting for conversation structure
    - Memory-efficient tensor preparation

    Attributes:
        tokenizer (PreTrainedTokenizer): HuggingFace tokenizer with ChatML support.
        max_length (int): Maximum sequence length for processing conversations.
        padding (int): Padding token ID for sequence length normalization.
        bos_id (List[int]): Token IDs for assistant start marker '<|im_start|>assistant'.
        eos_id (List[int]): Token IDs for conversation end marker '<|im_end|>'.
        data (List[Dict]): Loaded preference pair samples from JSONL file.
    """
    def __init__(self, file_path, tokenizer, max_length=4096):
        """
        Initialize the DPODataset.

        Args:
            file_path (str): Path to JSONL file containing preference pair data.
                           Each line should contain 'chosen' and 'rejected' conversation pairs.
            tokenizer (PreTrainedTokenizer): HuggingFace tokenizer with ChatML template support.
            max_length (int, optional): Maximum sequence length for conversation processing.
                                      Defaults to 4096 for longer conversations.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Handle padding token (fallback to 0 if not defined)
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
        # Extract special token sequences for loss mask generation
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids
        
        # Load preference pair data
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = []
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                self.data.append(obj)

    def __len__(self):
        """
        Return the total number of preference pairs.

        Returns:
            int: Number of preference pair samples available for DPO training.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Process and return a preference pair sample for DPO training.

        This method converts chosen and rejected conversation pairs into tensors suitable
        for direct preference optimization, with appropriate loss masking for each response.

        Args:
            index (int): Index of the preference pair to retrieve.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing training tensors:
                - x_chosen (torch.Tensor): Input tokens for chosen response. Shape: (max_length-1,).
                - y_chosen (torch.Tensor): Target tokens for chosen response. Shape: (max_length-1,).  
                - mask_chosen (torch.Tensor): Loss mask for chosen response. Shape: (max_length-1,).
                - x_rejected (torch.Tensor): Input tokens for rejected response. Shape: (max_length-1,).
                - y_rejected (torch.Tensor): Target tokens for rejected response. Shape: (max_length-1,).
                - mask_rejected (torch.Tensor): Loss mask for rejected response. Shape: (max_length-1,).

        **Processing Steps:**
        1. Extract chosen and rejected conversation pairs
        2. Format both conversations using ChatML template
        3. Tokenize and apply padding/truncation consistently
        4. Generate loss masks for both responses
        5. Create input-output pairs for autoregressive training
        6. Return structured dictionary for DPO loss computation

        **Tensor Alignment:**
        All tensors have consistent shapes to enable batch processing and loss computation.
        The masks ensure training focuses only on assistant responses in both chosen and rejected pairs.
        """
        item = self.data[index]
        
        # Extract preference pair conversations
        # chosen (List[Dict]): Preferred conversation with role-content structure
        chosen = item['chosen']  
        # rejected (List[Dict]): Rejected conversation with same structure
        rejected = item['rejected']
        
        # Format conversations using ChatML template
        # chosen_prompt (str): Formatted chosen conversation with proper role markers
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        # rejected_prompt (str): Formatted rejected conversation with proper role markers
        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        
        # Tokenize both conversations with consistent length constraints
        # chosen_encoding (transformers.BatchEncoding): Tokenized chosen conversation
        chosen_encoding = self.tokenizer(
            chosen_prompt, 
            truncation=True, 
            max_length=self.max_length, 
            padding='max_length'
        )
        # rejected_encoding (transformers.BatchEncoding): Tokenized rejected conversation
        rejected_encoding = self.tokenizer(
            rejected_prompt, 
            truncation=True, 
            max_length=self.max_length, 
            padding='max_length'
        )

        # Extract token IDs from encodings
        chosen_input_ids = chosen_encoding['input_ids']
        rejected_input_ids = rejected_encoding['input_ids']
        
        # Generate loss masks for selective training on assistant responses
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)
        
        # Create autoregressive training pairs for both chosen and rejected responses
        # Shift sequences by one position: X[0:n-1] predicts Y[1:n]
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)     # Input tokens
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)      # Target tokens
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)   # Aligned loss mask
        
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)   # Input tokens
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)    # Target tokens  
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long) # Aligned loss mask

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def _generate_loss_mask(self, input_ids):
        """
        Generate loss mask to train only on assistant responses.

        This method creates a binary mask identical to the SFTDataset implementation,
        ensuring consistent masking behavior across different training stages.

        Args:
            input_ids (List[int]): Tokenized conversation as list of token IDs.

        Returns:
            List[int]: Binary loss mask with same length as input_ids.
                      1 for assistant response tokens, 0 for user inputs and padding.

        **Implementation Note:**
        This method duplicates the logic from SFTDataset._generate_loss_mask to maintain
        consistency in loss masking across different training paradigms.
        """
        # Initialize all positions as masked (0)
        loss_mask = [0] * len(input_ids)
        i = 0
        
        # Scan through token sequence to find assistant response regions
        while i < len(input_ids):
            # Check if current position starts an assistant response marker
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                # Found assistant start marker, find the content region
                start = i + len(self.bos_id)  # Position after '<|im_start|>assistant'
                end = start
                
                # Search for the corresponding end marker
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                
                # Mark the assistant response region as trainable
                # Include positions from after BOS to after EOS (inclusive)
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                
                # Move index past the processed region
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
                
        return loss_mask


class RLAIFDataset(Dataset):
    """
    Dataset class for Reinforcement Learning from AI Feedback (RLAIF) training.

    This dataset processes conversation data for training models using AI-generated feedback
    and rankings. Unlike traditional RLHF that uses human feedback, RLAIF leverages AI systems
    to provide feedback, making it more scalable and cost-effective for large-scale training.

    **Training Objective:**
    Learn to generate high-quality responses based on AI feedback and rankings, improving
    model performance through reinforcement learning signals derived from AI evaluations.

    **Data Format:**
    Expects JSONL files with conversation structures similar to SFT, but optimized for
    prompt-answer separation to facilitate policy gradient training:
    ```json
    {
      "conversations": [
        {"content": "Explain the theory of relativity"},
        {"content": "Einstein's theory of relativity consists of two parts..."}
      ]
    }
    ```

    **RLAIF vs RLHF:**
    - **RLHF**: Uses human preferences and feedback for training
    - **RLAIF**: Uses AI-generated preferences and feedback, more scalable
    - **Data Processing**: Similar structure but optimized for policy training

    **Key Differences from SFT:**
    - Separates prompts from answers for independent processing
    - Designed for policy gradient training rather than supervised learning
    - Focuses on prompt-completion pairs rather than full conversations

    **Applications:**
    - Constitutional AI training
    - Self-improvement through AI feedback
    - Scalable preference learning without human annotation

    Attributes:
        tokenizer (PreTrainedTokenizer): HuggingFace tokenizer with ChatML template support.
        max_length (int): Maximum sequence length for conversation processing.
        samples (List[Dict]): Loaded conversation samples from JSONL file.
        bos_id (List[int]): Token IDs for assistant start marker '<|im_start|>assistant'.
        eos_id (List[int]): Token IDs for conversation end marker '<|im_end|>'.
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        """
        Initialize the RLAIFDataset.

        Args:
            jsonl_path (str): Path to JSONL file containing conversation data for RLAIF training.
                            Each line should contain a JSON object with 'conversations' field.
            tokenizer (PreTrainedTokenizer): HuggingFace tokenizer with ChatML template support.
                                           Must have apply_chat_template method available.
            max_length (int, optional): Maximum sequence length for conversation processing.
                                      Defaults to 1024 for efficient processing.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        
        # Extract special token sequences (same as SFT for consistency)
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids

    def __len__(self):
        """
        Return the total number of conversation samples.

        Returns:
            int: Number of conversation samples available for RLAIF training.
        """
        return len(self.samples)

    def load_data(self, path):
        """
        Load conversation data from JSONL file.

        Args:
            path (str): Path to the JSONL file containing conversation data.

        Returns:
            List[Dict]: List of conversation samples, each containing 'conversations' field
                       with conversation turns for RLAIF processing.

        Raises:
            FileNotFoundError: If the specified file path does not exist.
            json.JSONDecodeError: If any line contains invalid JSON.
            KeyError: If conversation structure is missing required fields.
        """
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        """
        Convert conversation into separate prompt and answer components.

        This method processes conversation turns and extracts the final assistant response
        as a separate answer, while formatting the preceding context as a prompt. This
        separation is crucial for RLAIF training where prompts and completions are
        processed independently.

        Args:
            conversations (List[Dict]): List of conversation turns, each containing 'content' field.
                                      Must have at least one turn for proper processing.

        Returns:
            Tuple[str, str]: A tuple containing:
                - prompt (str): Formatted prompt with conversation context up to the last user turn.
                               Includes generation prompt marker for completion.
                - answer (str): The final assistant response content as a standalone string.

        **Processing Logic:**
        1. Build conversation history with alternating user/assistant roles
        2. Extract the final assistant response as the target answer
        3. Format the context up to the last user turn as the prompt
        4. Add generation prompt to indicate where completion should begin

        **Example:**
        ```
        Input: [
          {"content": "What is AI?"},
          {"content": "AI is artificial intelligence..."},
          {"content": "Give an example"},
          {"content": "ChatGPT is an example of AI"}
        ]
        
        Output:
        prompt = "<|im_start|>user\nWhat is AI?<|im_end|>\n<|im_start|>assistant\nAI is...<|im_end|>\n<|im_start|>user\nGive an example<|im_end|>\n<|im_start|>assistant\n"
        answer = "ChatGPT is an example of AI"
        ```
        """
        # Build structured conversation with explicit roles
        messages = []
        answer = ''  # Will store the final assistant response
        
        for i, turn in enumerate(conversations):
            # Alternate between user and assistant roles based on turn index
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
            # Capture the last assistant response as the target answer
            answer = turn['content']
        
        # Create prompt from conversation context excluding the final assistant response
        # This provides the context for completion generation
        prompt = self.tokenizer.apply_chat_template(
            messages[:-1],                     # Exclude the final assistant response
            tokenize=False,                    # Return string format
            add_generation_prompt=True         # Add prompt for completion generation
        )
        
        return prompt, answer

    def __getitem__(self, index):
        """
        Process and return a conversation sample for RLAIF training.

        This method converts a conversation into prompt-answer pairs suitable for reinforcement
        learning from AI feedback. Unlike supervised fine-tuning, it separates the prompt context
        from the target completion to enable policy gradient training.

        Args:
            index (int): Index of the conversation sample to retrieve.

        Returns:
            Dict[str, str]: Dictionary containing training components:
                - prompt (str): Formatted conversation context up to the generation point.
                               Ready for model input to generate completions.
                - answer (str): Target response content for comparison and feedback generation.
                               Used as reference for AI feedback and policy updates.

        **Use in RLAIF Training:**
        1. **Prompt**: Fed to the model to generate multiple candidate responses
        2. **Answer**: Used as reference for AI feedback system to rank/score candidates
        3. **Feedback**: AI system compares generated responses against the reference answer
        4. **Policy Update**: Model parameters updated based on AI-generated feedback

        **Key Differences from SFT:**
        - Returns strings instead of tokenized tensors for flexible processing
        - Separates prompt and answer for independent manipulation
        - Optimized for generation and evaluation rather than direct supervision
        """
        sample = self.samples[index]
        
        # Extract prompt and answer components from conversation
        # prompt (str): Conversation context formatted for completion generation
        # answer (str): Target response content for reference and feedback
        prompt, answer = self._create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': answer
        }


if __name__ == "__main__":
    pass
