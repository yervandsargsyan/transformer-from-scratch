from tokenizer.bpe_trainer import BPETrainer

def test_encode_decode():
    text = "hello hello world"
    trainer = BPETrainer()
    tokenizer = trainer.train(text, vocab_size=300)

    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    # assert encode + decode returns original text
    assert decoded == text, f"Decoded text mismatch: {decoded} != {text}"

    # optional: check that merges were created
    assert len(tokenizer.merges) > 0, "No merges created"

    print("Test passed!")
