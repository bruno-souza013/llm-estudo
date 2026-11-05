# LLM Zero

Um mini modelo de linguagem em português, treinado do zero com artigos da
Wikipédia em português. Este projeto acompanha o processo de implementação de
um modelo estilo GPT para fins de estudo.

> **Status:** primeira versão funcional.

O pipeline completo já funciona:

```text
Wikipédia em português
        ↓
preparação do dataset
        ↓
tokenização GPT-2
        ↓
arquivos train.bin e val.bin
        ↓
Transformer causal
        ↓
treinamento em GPU
        ↓
checkpoint
        ↓
geração de texto
```

## Resultado da primeira versão

Nesta versão, o projeto:

- baixa artigos da Wikipédia em português usando streaming;
- cria automaticamente `data/wiki_pt/`;
- divide os dados em treino e validação;
- usa o tokenizer BPE do GPT-2 por meio do `tiktoken`;
- salva os tokens em arquivos binários;
- implementa um Transformer causal inspirado no nanoGPT;
- treina com AdamW e salva o melhor checkpoint pela `val loss`;
- gera texto a partir de `out/ckpt.pt`;
- pode ser executado em CPU ou em uma GPU CUDA, como as disponíveis no Kaggle.

O primeiro modelo conseguiu gerar texto em português com palavras, acentos e
estruturas semelhantes às do corpus. A coerência ainda é limitada porque o
dataset e o modelo são pequenos. Esse é um resultado esperado para uma
primeira versão didática.

## Estrutura

```text
llm-zero/
├── config.py              # configurações do modelo e do treinamento
├── model.py               # Transformer causal e geração
├── train.py               # carregamento dos dados e loop de treino
├── generate.py            # carregamento do checkpoint e geração
├── data/
│   ├── prepare.py         # download, tokenização e criação dos binários
│   └── wiki_pt/
│       ├── train.bin
│       └── val.bin
├── out/
│   └── ckpt.pt            # criado durante o treinamento
└── README.md
```

## Requisitos

- Python 3.10 ou superior;
- PyTorch;
- NumPy;
- `tiktoken`;
- `datasets`;
- `tqdm`;
- GPU CUDA opcional.

Instalação:

```powershell
python -m pip install torch numpy tiktoken datasets tqdm
```

Em um ambiente virtual do Windows:

```powershell
.\venv\Scripts\python.exe -m pip install torch numpy tiktoken datasets tqdm
```

## Preparando os dados

O arquivo [data/prepare.py](data/prepare.py) cria a pasta `data/wiki_pt` e
salva os arquivos binários usados no treinamento.

Execute:

```powershell
.\venv\Scripts\python.exe data\prepare.py
```

A quantidade de artigos é definida em `QUANTIDADE_ARTIGOS`:

```python
QUANTIDADE_ARTIGOS = 2_000
```

O script só baixa os artigos quando `data/input.txt` ainda não existe. Para
baixar uma nova quantidade, remova o arquivo anterior antes de executar:

```powershell
Remove-Item .\data\input.txt
.\venv\Scripts\python.exe .\data\prepare.py
```

Os arquivos gerados são:

```text
data/wiki_pt/train.bin
data/wiki_pt/val.bin
```

## Configuração

As configurações ficam em [config.py](config.py).

Exemplo de configuração pequena:

```python
class ModelConfig:
    vocab_size = 50257
    block_size = 128
    n_layer = 4
    n_head = 4
    n_embd = 128
    dropout = 0.0
    bias = True
```

Para usar uma GPU CUDA:

```python
device = "cuda"
```

Para executar em CPU:

```python
device = "cpu"
```

## Treinamento

Depois de preparar os dados:

```powershell
.\venv\Scripts\python.exe train.py
```

Durante o treinamento, são exibidas as perdas de treino e validação:

```text
step 0: train loss ..., val loss ...
step 500: train loss ..., val loss ...
```

O melhor checkpoint é salvo em:

```text
out/ckpt.pt
```

O arquivo de checkpoint contém:

- pesos do modelo;
- configuração da arquitetura;
- configuração do treino;
- passo salvo;
- menor `val_loss` observada.

## Geração de texto

Coloque o checkpoint em `out/ckpt.pt` e execute:

```powershell
.\venv\Scripts\python.exe generate.py
```

O [generate.py](generate.py) usa o mesmo tokenizer GPT-2 empregado na
preparação dos dados. Isso é necessário para que os IDs dos tokens sejam
compatíveis com os pesos treinados.

Os parâmetros de geração podem ser ajustados, por exemplo:

```python
temperature=0.6
top_k=50
```

Temperaturas menores geralmente produzem saídas menos aleatórias, mas não
substituem um dataset maior ou um treinamento mais longo.

## Treinamento no Kaggle

O projeto pode ser clonado em um Notebook do Kaggle com uma GPU ativada:

```python
!git clone https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
%cd SEU_REPOSITORIO
!python train.py
```

Confirme o dispositivo:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

O checkpoint criado pelo Kaggle pode ser salvo como output da versão do
Notebook e baixado para a pasta `out/` localmente.

## Limitações conhecidas

- O corpus da primeira versão é pequeno para um modelo de linguagem geral.
- O tokenizer `gpt2` não foi treinado especificamente para português.
- O modelo pode memorizar partes da Wikipédia e apresentar overfitting.
- A geração ainda pode produzir frases sem coerência.
- O checkpoint não deve ser versionado no Git; ele é ignorado pelo
  `.gitignore`.

## Próximos passos

- aumentar e limpar o corpus;
- adicionar um token explícito de fim de documento entre artigos;
- comparar mais passos de treinamento;
- ajustar `batch_size`, `block_size` e o tamanho do modelo;
- adicionar geração interativa por prompt;
- registrar métricas e comparar checkpoints;
- avaliar um tokenizer treinado para português;
- criar testes automatizados para o modelo e o carregamento de dados.

## Créditos e referências

Este projeto é uma implementação educacional própria, inspirada na arquitetura
e no fluxo de treinamento do **nanoGPT**, criado por **Andrej Karpathy**:

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- [Andrej Karpathy](https://github.com/karpathy)

O nanoGPT é usado aqui como referência de estudo.

Fontes e ferramentas utilizadas:

- **Wikimedia Foundation** — conteúdo da Wikipédia e dumps da Wikimedia:
  [Wikimedia Downloads](https://dumps.wikimedia.org/)
- **Hugging Face** — biblioteca Datasets e o dataset
  [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia)
- **OpenAI** — biblioteca [tiktoken](https://github.com/openai/tiktoken) e
  codificação GPT-2 utilizada neste experimento;
- **PyTorch** — framework utilizado para implementar e treinar o modelo:
  [pytorch.org](https://pytorch.org/).

Consulte as licenças e os termos de uso de cada fonte antes de redistribuir o
dataset ou publicar um modelo treinado com esses dados.

