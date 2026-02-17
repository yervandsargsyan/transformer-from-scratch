import torch
from model.transformer import Transformer
from scripts.load_tokenizer import load_tokenizer

def generate_text(model, tokenizer, seed_text="<bos>", max_len=500, device="cpu", temperature=1.0, top_p=0.9):
    """
    Generate text using the trained Transformer with nucleus (top-p) sampling.

    - temperature: controls randomness (higher = more diverse)
    - top_p: cumulative probability threshold (0.9 = sample from tokens that make 90% of probability mass)
    """
    # Initialize tokens
    tokens = [tokenizer.special_tokens[seed_text]] if seed_text in tokenizer.special_tokens else [tokenizer.special_tokens["<bos>"]]
    generated = tokens.copy()

    for _ in range(max_len):
        x = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(device)  # [1, seq_len]
        with torch.no_grad():
            logits = model(x)[0, -1] / temperature  # [vocab_size]
            probs = torch.softmax(logits, dim=-1)

        # --- Top-p (nucleus) sampling ---
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)

        mask = cumulative_probs <= top_p
        if not mask.any():
            mask[0] = True
        filtered_probs = sorted_probs[mask]
        filtered_probs /= filtered_probs.sum()  # renormalize
        filtered_indices = sorted_indices[mask]

        next_id = filtered_indices[torch.multinomial(filtered_probs, 1)].item()
        generated.append(next_id)

        if next_id == tokenizer.special_tokens["<eos>"]:
            break

    text = tokenizer.decode(generated)
    return text

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    tokenizer = load_tokenizer("tokenizer.json")

    # Load model once
    model = Transformer(vocab_size=len(tokenizer.vocab)).to(device)
    model.load_state_dict(torch.load("transformer_epoch10.pth", map_location=device))
    model.eval()  # set evaluation mode

    # Generate long text
    text = generate_text(
        model,
        tokenizer,
        seed_text="<bos>",
        max_len=500,
        device=device,
        temperature=1.0,
        top_p=0.9
    )

    print("Generated text:\n")
    print(text)

if __name__ == "__main__":
    main()
