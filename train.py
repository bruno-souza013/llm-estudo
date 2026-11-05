import math
import os
from contextlib import nullcontext

import numpy as np
import torch

from config import ModelConfig, TrainConfig
from model import GPT


def get_batch(
    split: str,
    data_config: TrainConfig,
    model_config: ModelConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    filename = "train.bin" if split == "train" else "val.bin"
    path = os.path.join(data_config.data_dir, filename)
    data = np.memmap(path, dtype=np.uint16, mode="r")

    required_tokens = model_config.block_size + 1
    if len(data) < required_tokens:
        raise ValueError(
            f"{path} precisa ter pelo menos {required_tokens} tokens; "
            f"encontrados {len(data)}"
        )

    starts = torch.randint(
        len(data) - model_config.block_size,
        (data_config.batch_size,),
    )
    x = torch.stack(
        [
            torch.from_numpy(
                data[start : start + model_config.block_size].astype(np.int64)
            )
            for start in starts
        ]
    )
    y = torch.stack(
        [
            torch.from_numpy(
                data[start + 1 : start + 1 + model_config.block_size].astype(
                    np.int64
                )
            )
            for start in starts
        ]
    )
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    data_config: TrainConfig,
    model_config: ModelConfig,
    device: torch.device,
) -> dict[str, float]:
    result: dict[str, float] = {}
    model.eval()

    for split in ("train", "val"):
        losses = torch.zeros(data_config.eval_iters)
        for iteration in range(data_config.eval_iters):
            x, y = get_batch(split, data_config, model_config, device)
            _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("O modelo não retornou loss durante a avaliação")
            losses[iteration] = loss.item()
        result[split] = losses.mean().item()

    model.train()
    return result


def main() -> None:
    model_config = ModelConfig()
    train_config = TrainConfig()

    if train_config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "TrainConfig.device está como 'cuda', mas CUDA não está disponível. "
            "Altere para device='cpu' em config.py."
        )

    device = torch.device(train_config.device)
    torch.manual_seed(train_config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(train_config.seed)

    os.makedirs(train_config.out_dir, exist_ok=True)
    model = GPT(model_config).to(device)
    print(f"Parâmetros: {model.num_parameters():,}")
    print(f"Dispositivo: {device}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        weight_decay=train_config.weight_decay,
    )
    context = (
        nullcontext()
        if device.type == "cpu"
        else torch.autocast(device_type=device.type, dtype=torch.float16)
    )
    best_val_loss = math.inf

    for iteration in range(train_config.max_iters):
        if iteration % train_config.eval_interval == 0:
            losses = estimate_loss(model, train_config, model_config, device)
            print(
                f"step {iteration}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "model_config": model_config.__dict__,
                        "train_config": train_config.__dict__,
                        "step": iteration,
                        "val_loss": best_val_loss,
                    },
                    os.path.join(train_config.out_dir, "ckpt.pt"),
                )

        optimizer.zero_grad(set_to_none=True)
        for _ in range(train_config.gradient_accumulation_steps):
            x, y = get_batch("train", train_config, model_config, device)
            with context:
                _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("O modelo não retornou loss durante o treino")
            (loss / train_config.gradient_accumulation_steps).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if iteration % 10 == 0:
            print(f"step {iteration}: batch loss {loss.item():.4f}")


if __name__ == "__main__":
    main()