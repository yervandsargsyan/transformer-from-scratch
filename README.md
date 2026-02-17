# transformer-from-scratch

**Pure PyTorch GPT-style Transformer from scratch**  
Includes byte-level BPE tokenizer, causal self-attention, training loop, and autoregressive text generation.

---

##  Features

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
