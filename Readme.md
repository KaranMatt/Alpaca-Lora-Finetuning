# Alpaca Fine-tuning with Qwen2-0.5B

A lightweight implementation of instruction fine-tuning using the Alpaca dataset format with the Qwen2-0.5B model, optimized with LoRA and 8-bit quantization.

> **Fully Local Training**: This project runs entirely on consumer-grade GPUs with VRAM constraints in mind. The model is installed and trained locally without requiring cloud infrastructure or high-end hardware.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Configuration](#training-configuration)
- [Training Metrics Explained](#training-metrics-explained)
- [Usage](#usage)
- [Results](#results)
- [File Structure](#file-structure)

## Overview

This project demonstrates efficient fine-tuning of the Qwen2-0.5B instruction-following model using Parameter-Efficient Fine-Tuning (PEFT) techniques. The implementation uses the Alpaca dataset format with LoRA adapters and 8-bit quantization to reduce memory requirements while maintaining model quality.

**Local & Accessible**: The entire pipeline runs on consumer-grade graphics cards, making advanced AI fine-tuning accessible without expensive cloud GPUs or enterprise hardware. All models and datasets are downloaded and stored locally.

## Features

- **Fully Local Training**: Complete pipeline runs on consumer GPUs (tested on standard gaming hardware)
- **VRAM Optimized**: Designed for 6-8GB VRAM constraints using aggressive optimization
- **Offline Capable**: All models and datasets stored locally, no internet required after initial download
- **Memory Efficient**: Uses 8-bit quantization via BitsAndBytes (~75% memory reduction)
- **Parameter Efficient**: LoRA fine-tuning with only ~0.3% trainable parameters
- **Instruction Format**: Alpaca-style instruction-input-output format
- **Optimized Training**: Gradient accumulation and mixed precision training
- **Ready for Deployment**: Trained adapters can be merged or loaded separately

## Requirements

```python
pandas
datasets
transformers
peft
torch
bitsandbytes
trl
```

### Hardware Requirements

**Minimum:**
- GPU: NVIDIA GTX 1660 / RTX 2060 or equivalent (6GB VRAM)
- RAM: 16GB system memory
- Storage: 10GB free space for models and data

**Recommended:**
- GPU: RTX 3060 / RTX 4060 or better (8GB+ VRAM)
- RAM: 32GB system memory
- Storage: 20GB free space

**Tested On:**
- Consumer-grade gaming GPUs
- Windows/Linux environments
- Local Python virtual environment

> **Note**: This project is specifically optimized for local training on consumer hardware. No cloud GPUs or enterprise infrastructure required!

## Dataset

The project uses the Alpaca dataset format with 52,002 instruction-following examples, available in multiple formats for flexibility.

**Dataset Files:**
- `Finance Alpaca.parquet` - Original Parquet format (efficient storage)
- `Alp_csv.csv` - CSV format (same dataset, human-readable)

**Dataset Structure:**
- `instruction`: The task description
- `input`: Additional context (optional)
- `output`: Expected response

**Preprocessing Steps:**
1. Handle missing values in input field (filled with empty strings)
2. Remove entries with missing outputs (31 removed)
3. Sample 5,000 examples for efficient training
4. 80/20 train-test split (4,000 train / 1,000 test)

**Example Format:**
```
Instruction: Write a tweet about AI technology.
Input: 
Output: AI is transforming the way we work, learn, and...
```

## Model Architecture

**Base Model:** Qwen2-0.5B-Instruct
- Lightweight 500M parameter model
- Optimized for instruction following
- Supports multiple languages
- **Locally installed** via Hugging Face transformers

**Quantization:**
- 8-bit quantization via BitsAndBytes
- Reduces memory footprint by ~75%
- Minimal accuracy degradation
- **Enables training on consumer GPUs with 6-8GB VRAM**

**LoRA Configuration:**
```python
Rank (r): 8
Alpha: 16
Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
Dropout: 0.1
Task: Causal Language Modeling
```

### Memory Optimization Strategies

This implementation employs multiple techniques to fit within consumer GPU constraints:

1. **8-bit Quantization**: Reduces model size from ~2GB to ~500MB
2. **LoRA Adapters**: Only trains ~0.3% of parameters instead of full model
3. **Gradient Accumulation**: Simulates larger batch sizes without memory overhead
4. **Small Batch Size**: 2 samples per device with 4-step accumulation
5. **Sequence Truncation**: Max length of 512 tokens
6. **Paged Optimizer**: `paged_adamw_8bit` reduces optimizer memory by 50%

**Result**: Training runs smoothly on GPUs with as little as 6GB VRAM!

## Training Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Epochs** | 2 | Number of complete passes through dataset |
| **Batch Size** | 2 per device | Samples processed simultaneously |
| **Gradient Accumulation** | 4 steps | Effective batch size = 8 |
| **Learning Rate** | 1e-5 | Step size for weight updates |
| **Max Sequence Length** | 512 tokens | Maximum input/output length |
| **Optimizer** | paged_adamw_8bit | Memory-efficient Adam variant |
| **Precision** | BF16 | Brain Float 16 for stability |
| **Save Strategy** | Every 200 steps | Checkpoint frequency |

## Training Metrics Explained

### Final Training Results

| Epoch | Training Loss | Validation Loss | Entropy | Mean Token Accuracy |
|-------|---------------|-----------------|---------|---------------------|
| 1 | 1.4077 | 1.3905 | 1.3932 | 66.70% |
| 2 | 1.3700 | 1.3833 | 1.3761 | 66.82% |

### Metric Significance

#### **Training Loss: 1.3700**
The average error during training across all batches.
- **What it means**: Lower is better; measures how well the model predicts the next token
- **Trend**: Decreased from 1.4077 → 1.3700, indicating effective learning
- **Good sign**: Consistent decrease shows the model is improving

#### **Validation Loss: 1.3833**
Error on unseen test data, measuring generalization ability.
- **What it means**: How well the model performs on new examples
- **Trend**: Slight decrease from 1.3905 → 1.3833
- **Interpretation**: Very close to training loss (minimal overfitting)
- **Key insight**: The model generalizes well to new instructions

#### **Entropy: 1.3761**
Measures prediction uncertainty/randomness.
- **What it means**: Lower entropy = more confident predictions
- **Range**: Typically 0 (certain) to log(vocab_size) (random)
- **Trend**: Decreased from 1.3932 → 1.3761
- **Good sign**: Model is becoming more confident in its predictions

#### **Mean Token Accuracy: 66.82%**
Percentage of correctly predicted tokens.
- **What it means**: Direct measure of prediction accuracy
- **Interpretation**: ~67% of tokens are predicted correctly
- **Context**: This is strong for instruction-following tasks
- **Note**: Includes all tokens (articles, prepositions, etc.), not just meaningful words

### Overall Assessment

**Strengths:**
-  Minimal overfitting (training/validation loss very close)
-  Consistent improvement across all metrics
-  Good generalization to new examples
-  Increasing prediction confidence

**What the metrics tell us:**
1. The model learned the instruction-following pattern effectively
2. It generalizes well without memorizing training data
3. Predictions are becoming more confident and accurate
4. The model is ready for inference on new instructions

## Usage

### Initial Setup

1. **Clone the repository**
   ```bash
   git clone <https://github.com/KaranMatt/Alpaca-Lora-Finetuning.git>
   cd <Alpaca-Lora-Finetuning>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the dataset**
   - Dataset files are already included in `data/` directory
   - Choose either `Finance Alpaca.parquet` or `Alp_csv.csv`
   - Both contain identical data in different formats

### Training

```python
# Open and run the notebook
jupyter notebook "Alpaca Finetune.ipynb"

# Or run all cells programmatically
# The notebook handles:
# - Data loading and preprocessing
# - Model initialization with 8-bit quantization
# - LoRA configuration
# - Training loop
# - Checkpoint saving
```

### Loading the Fine-tuned Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = "Qwen/Qwen2-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(base_model)
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Load fine-tuned LoRA adapters
model = PeftModel.from_pretrained(model, "qwen2_lora_out/checkpoint-1000")

# Use config.json for model settings
# The config file contains all architecture parameters
```

### Key Functions

**Data Formatting:**
```python
def format_prompt(ex):
    """Converts Alpaca format to chat template"""
    instruction = ex['instruction']
    context = ex['input']
    response = ex['output']
    
    if str(context).strip():
        full_instruction = f'{instruction}\n Context:{context}'
    else:
        full_instruction = instruction
    
    messages = [
        {'role': 'user', 'content': full_instruction},
        {'role': 'assistant', 'content': response}
    ]
    
    return {'text': tokenizer.apply_chat_template(messages, tokenize=False)}
```

## File Structure

```
project/
│
├── Alpaca Finetune.ipynb    # Main training notebook
│
├── config.json              # Fine-tuned model configuration
│                            # Contains model architecture and settings
│
├── data/
│   ├── Finance Alpaca.parquet   # Alpaca dataset (Parquet format)
│   └── Alp_csv.csv              # Alpaca dataset (CSV format - same data)
│
└── qwen2_lora_out/          # Training output directory
    ├── checkpoint-200/       # Saved checkpoints every 200 steps
    ├── checkpoint-800/
    └── checkpoint-1000/      # Final model checkpoint
        ├── adapter_config.json
        ├── adapter_model.safetensors
        └── ...
```

### Important Files

**config.json**
- Contains the configuration of the fine-tuned model
- Includes model architecture parameters
- Essential for loading and using the trained model
- Auto-generated after training completion

**Dataset Files**
- Both Parquet and CSV formats contain identical data
- Parquet: More efficient for large-scale processing
- CSV: Human-readable, easier to inspect and modify
- Use either format based on your preference

## Results

**Training Completed Successfully:**
- Total training time: ~46 minutes
- Training samples per second: 2.866
- Final training loss: 1.4447
- Final validation accuracy: 66.82%

**Model Checkpoints:**
- Saved every 200 steps
- Best checkpoint can be loaded for inference
- LoRA adapters are lightweight (~few MB)

## Next Steps

1. **Inference**: Load the trained adapter for generating responses
2. **Merge Adapters**: Combine LoRA weights with base model
3. **Evaluation**: Test on custom instruction datasets
4. **Deployment**: Export for production use
5. **Fine-tuning**: Adjust hyperparameters for specific domains

## Notes

- The model uses chat templates for proper formatting
- Padding is set to `right` with EOS token as pad token
- All target transformer layers are fine-tuned via LoRA
- The implementation prioritizes efficiency over maximum accuracy

## Contributing

Feel free to:
- Experiment with different hyperparameters
- Try different base models
- Extend the dataset
- Optimize for specific use cases

## License

This project uses models and libraries with their respective licenses. Please check individual component licenses before commercial use.

---

**Built with** Transformers | PEFT | BitsAndBytes | TRL