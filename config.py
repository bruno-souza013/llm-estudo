from dataclasses import dataclass

@dataclass
class ModelConfig:
    #vocabulario do tokenizer gpt2
    vocab_size: int = 50257
    
    #tamanho maximo do contexto
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
    
    #checkpoints
    out_dir: str = "out"
    
    #dispositivo: cpu, cuda ou cuda0
    device: str = "cuda"
    
    #dados por lote
    batch_size: int = 2

    #quantidade de passos de acumulação de gradiente
    gradient_accumulation_steps: int = 1

    #quantidade total de iterações
    max_iters: int = 1000

    #frequência de avaliação
    eval_interval: int = 100
    eval_iters: int = 20

    #taxa de aprendizado
    learning_rate: float = 3e-4

    #regularização
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95

    #semente para resultados reproduzíveis
    seed: int = 1337