# 🚀 Small Language Model Training

This project implements a **125M parameter language model** optimized for training on consumer hardware with limited VRAM (4GB+). It includes efficient training with gradient accumulation and length-based batch scheduling.

## 📂 Project Structure
```
│── model.py        # Transformer-based language model (125M params)
│── train.py        # Training script with memory optimizations
│── inference.py    # Text generation script
│── requirements.txt # Required dependencies
│── README.md       # Project documentation
```

## 📌 Features
- **Memory-Efficient Transformer Model** (~125M parameters)
- **Length-Based Batch Scheduling** for efficient training
- **Gradient Accumulation** for effective larger batch sizes
- **Autoregressive Text Generation**
- **Wikitext-2 Dataset Integration**

## 🛠 Installation
Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Training the Model
Run the training script:
```bash
python train.py
```

The training process includes:
- Automatic GPU/CPU device selection
- Dynamic batch scheduling by sequence length
- Gradient accumulation (effective batch size: 16)
- Automatic checkpointing
- Cosine learning rate scheduling

## 📝 Inference
Generate text using the trained model:
```bash
python inference.py
```

## 🏗 Model Architecture
- **Layers:** 12 transformer blocks
- **Attention Heads:** 12 heads
- **Embedding Dimension:** 768
- **Context Window:** 512 tokens
- **Total Parameters:** ~125M
- **Activation:** GELU
- **Layer Normalization:** Pre-norm architecture

## ⚡ Performance Optimizations
- ✅ Length-based batch scheduling
- ✅ Gradient accumulation (4 steps)
- ✅ Efficient memory usage
- ✅ Optimized for 4GB VRAM GPUs
- ✅ Pre-padded sequences for faster training

## 🔧 Training Configuration
- **Batch Size:** 4 (16 with gradient accumulation)
- **Learning Rate:** 3e-4 with cosine decay
- **Weight Decay:** 0.1
- **Training Data:** Wikitext-2
- **Epochs:** 3

## 📊 Memory Usage
- **GPU VRAM:** ~3.5GB peak
- **Recommended GPU:** 4GB+ VRAM
- **CPU RAM:** ~8GB recommended

## 📜 License
This project is licensed under the MIT License.

---
🚀 Happy Training! Feel free to contribute or raise issues. 🎯


