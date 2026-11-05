from pathlib import Path

import torch
import tiktoken

from model import LLM
from config import ModelConfig


CHECKPOINT_PATH = Path("out/ckpt.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    model_config = ModelConfig(**checkpoint["model_config"])
    model = LLM(model_config)
    model.load_state_dict(checkpoint["model"])
    model.to(DEVICE)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")

    prompt = "Os jogadores brasileiros que participaram da Copa do Mundo"
    token_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(
        token_ids,
        dtype=torch.long,
        device=DEVICE,
    ).unsqueeze(0)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=200,
            temperature=0.8,
            top_k=100,
        )

    generated_text = tokenizer.decode(generated_ids[0].tolist())
    print(generated_text)


if __name__ == "__main__":
    main()