# 🚀 Small Language Model with LoRA Fine-Tuning

This project implements a **500M parameter language model** fine-tuned with **LoRA (Low-Rank Adaptation)** for efficient training on consumer hardware. It includes training, inference, and dataset preprocessing.

## 📂 Project Structure
```
│── model.py        # Defines the Transformer-based language model
│── train.py        # Trains the model using LoRA
│── inference.py    # Generates text from the trained model
│── requirements.txt # Required dependencies
│── README.md       # Project documentation
```

## 📌 Features
- **Lightweight Transformer Model**
- **LoRA Fine-Tuning** for reduced GPU memory usage
- **Autoregressive Text Generation**
- **Wikitext-2 Dataset Preprocessing**

## 🛠 Installation
First, install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Training the Model
Run the training script to fine-tune the model with LoRA:
```bash
python train.py
```
This will:
- Load a custom Transformer model.
- Apply LoRA to reduce trainable parameters.
- Train on the Wikitext-2 dataset.
- Save LoRA adapters in `lora_adapter/`.

## 📝 Inference (Text Generation)
After training, generate text with:
```bash
python inference.py
```
Example output:
```
Once upon a time, in a faraway kingdom...
```

## 🏗 Model Details
- **Transformer Architecture:** 6 layers, 8 heads
- **Embedding Size:** 512
- **LoRA Rank:** 8
- **Dataset:** Wikitext-2

## 📌 LoRA Benefits
✅ **Reduces GPU memory** usage
✅ **Faster fine-tuning**
✅ **Lightweight adaptation**

## 📜 License
This project is licensed under the MIT License.

---
🚀 Happy Fine-Tuning! Let me know if you need any improvements. 🎯


