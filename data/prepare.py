from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


DATA_DIR = Path(__file__).parent
INPUT_FILE = DATA_DIR / "input.txt"
OUTPUT_DIR = DATA_DIR / "wiki_pt"
QUANTIDADE_ARTIGOS = 2_000

#cria data/wiki_pt caso ela ainda não exista
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


#baixa 2.000 artigos somente se input.txt ainda não existir
if not INPUT_FILE.exists():
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.pt",
        split="train",
        streaming=True,
    )

    artigos_salvos = 0

    with INPUT_FILE.open("w", encoding="utf-8") as arquivo:
        for exemplo in tqdm(dataset, total=QUANTIDADE_ARTIGOS):
            texto = exemplo["text"].strip()

            if not texto:
                continue

            arquivo.write(texto)
            arquivo.write("\n\n")

            artigos_salvos += 1

            if artigos_salvos >= QUANTIDADE_ARTIGOS:
                break

    print(f"{artigos_salvos} artigos salvos em {INPUT_FILE}")


#lê o texto
with INPUT_FILE.open("r", encoding="utf-8") as arquivo:
    data = arquivo.read()

n = len(data)

#divide em 90% para treino e 10% para validação
train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9) :]


#tokenizador compativel com gpt2 bpe
encode = tiktoken.get_encoding("gpt2")

train_ids = encode.encode_ordinary(train_data)
val_ids = encode.encode_ordinary(val_data)

print(f"No treino há {len(train_ids):,} tokens")
print(f"Na validação há {len(val_ids):,} tokens")


#converte para inteiros de 16 bits
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)


#salva diretamente em data/wiki_pt/
train_file = OUTPUT_DIR / "train.bin"
val_file = OUTPUT_DIR / "val.bin"

train_ids.tofile(train_file)
val_ids.tofile(val_file)

print(f"Arquivo de treino salvo em: {train_file}")
print(f"Arquivo de validação salvo em: {val_file}")

#No treino há 2,083,350 tokens
#Na validacao há 229,806 tokens