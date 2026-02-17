# Transformer from Scratch

This project is an educational yet fully functional implementation of a GPT-style Transformer and BPE tokenizer written from scratch in PyTorch.

The main goal of the project is to help developers understand how Transformers and Large Language Models work internally, without hiding core logic behind high-level libraries and frameworks.
At the same time, this is not a toy example — the project provides a complete, working pipeline for tokenization, training, and text generation, suitable for small-scale experiments and research.

The code intentionally prioritizes clarity and transparency over performance optimizations, making it easy to read, modify, and experiment with.
This repository is designed both as a learning resource and as a solid foundation for further extensions.

**Key characteristics:**
- GPT-style decoder-only Transformer implemented from first principles
- Custom BPE tokenizer implemented without external NLP libraries
- End-to-end training and text generation pipeline
- Focus on understanding attention, embeddings, and training dynamics

**Important note:**
This project is not intended for large-scale pretraining or production deployment.
Its primary purpose is educational, while still remaining a fully operational and extensible implementation.

---

## Features

- Minimal GPT-style Transformer implemented fully in PyTorch  
- Byte-level BPE tokenizer for flexible tokenization  
- Causal self-attention (autoregressive)  
- Training loop with checkpointing per epoch  
- Autoregressive text generation with temperature and top‑p (nucleus) sampling  
- Lightweight and easy to extend  

---

##  Installation

```bash
git clone https://github.com/yervandsargsyan/transformer-from-scratch.git
cd transformer-from-scratch
pip install -r requirements.txt
```

---

## Training

### Prepare your corpus
Put your text in `data/corpus.txt`.  
You can use the included sample text or replace it with your own.

### Train the tokenizer (one-time)
Before training the model, you must first train the tokenizer on your corpus:

```bash
python -m scripts.train_tokenizer
```
### Train model
```bash
python -m scripts.train_loop     
```

---

## Generating
```bash
python -m scripts.generate_text
```
or
```bash
python -m scripts.generate_long_text
```
