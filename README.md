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
