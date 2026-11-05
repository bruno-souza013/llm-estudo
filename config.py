from dataclasses import dataclass


@dataclass
class ModelConfig:
 
    vocab_size: int = 50257

    #tamanho máximo do contexto
    block_size: int = 128

    #arquitetura do transformer
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True


@dataclass
class TrainConfig:
    data_dir: str = "data/wiki_pt"

    out_dir: str = "out"
    device: str = "cuda"
    batch_size: int = 32

    #passos de acumulação de gradiente
    #tokens por iteração = batch_size * gradient_accumulation_steps * block_size
    gradient_accumulation_steps: int = 2

    #quantidade total de iterações
    max_iters: int = 10000

    #frequência de avaliação
    eval_interval: int = 500
    eval_iters: int = 50


    learning_rate: float = 6e-4
    min_lr: float = 6e-5  # piso do decaimento cosseno (~ learning_rate / 10)
    warmup_iters: int = 200

    # regularização
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    resume: bool = False

    # semente para resultados reproduzíveis
    seed: int = 1337