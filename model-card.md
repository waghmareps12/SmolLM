---
language: en
tags:
- pytorch
- causal-lm
- language-model
- text-generation
license: mit
datasets:
- wikitext
metrics:
- perplexity
- loss
library_name: pytorch
pipeline_tag: text-generation

model-index:
- name: SmolLM-125M
  results:
  - task: 
      type: text-generation
      name: Language Modeling
    dataset:
      type: wikitext
      name: WikiText-2
    metrics:
      - type: perplexity
        value: to_be_updated  # Will be updated after training
      - type: loss
        value: to_be_updated  # Will be updated after training
---

# SmolLM-125M: A Lightweight Language Model for Consumer Hardware

This is a 125M parameter language model designed to be trained and run on consumer hardware with limited VRAM (4GB+). The model follows a GPT-style architecture but is optimized for efficiency and memory usage.

## Model Details

- **Architecture:** GPT-style Transformer
- **Parameters:** 125M
- **Context Length:** 512 tokens
- **Vocabulary:** 50,257 tokens (GPT-2 tokenizer)
- **Training Data:** WikiText-2
- **Hardware Requirements:** 4GB+ VRAM GPU

### Architecture Specifications
- Layers: 12 transformer blocks
- Attention Heads: 12
- Embedding Dimension: 768
- Activation: GELU
- Layer Normalization: Pre-norm

## Training Details

- **Hardware Used:** GTX 1650 (4GB VRAM)
- **Training Time:** ~4 hours
- **Batch Size:** 4 (16 with gradient accumulation)
- **Learning Rate:** 3e-4 with cosine decay
- **Weight Decay:** 0.1
- **Optimizer:** AdamW

### Memory Optimizations
1. Length-based batch scheduling
2. Gradient accumulation (4 steps)
3. Dynamic batch scheduling
4. Pre-padded sequences

## Usage

```python
from transformers import AutoTokenizer
from model import SmallLanguageModel, ModelConfig

# Initialize model
config = ModelConfig(
    vocab_size=50257,
    block_size=512,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.1,
    bias=True
)
model = SmallLanguageModel(config)

# Generate text
tokenizer = AutoTokenizer.from_pretrained("gpt2")
input_text = "Once upon a time"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
output_ids = model.generate(input_ids, max_length=100)
generated_text = tokenizer.decode(output_ids[0])
```

## Limitations

- Limited context window (512 tokens)
- Smaller capacity compared to larger models
- Training data limited to WikiText-2

## License
This model is released under the MIT License. 