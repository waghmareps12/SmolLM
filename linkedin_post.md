🚀 Excited to share my latest project: SmolLM-125M - A Small Language Model for Consumer Hardware!

I've implemented a GPT-style transformer from scratch using PyTorch, designed specifically for training on consumer GPUs. The model has 125M parameters and can run on graphics cards with just 4GB VRAM.

🛠️ Technical Highlights:
- 12-layer transformer (768 dim, 12 heads)
- Trains in ~4 hours on GTX 1650
- Uses only 3.5GB VRAM
- Trained on Wikitext-2 dataset

⚡ Key Optimizations:
1. Length-based sequence batching
2. Gradient accumulation
3. Memory-efficient attention
4. Pre-norm architecture

While not instruction-tuned yet, this implementation demonstrates how to make language model training accessible on modest hardware. Perfect for learning, experimentation, and building domain-specific models.

🔗 Try it yourself:
- GitHub: https://github.com/waghmareps12/SmolLM
- Hugging Face: https://huggingface.co/waghmareps12/SmolLM_125M

The repository includes:
- Complete PyTorch implementation
- Optimized training pipeline
- Detailed documentation
- Inference examples

#MachineLearning #AI #PyTorch #NLP #DeepLearning

P.S. Interested in training language models on consumer hardware? Let's connect and discuss! 🤝 