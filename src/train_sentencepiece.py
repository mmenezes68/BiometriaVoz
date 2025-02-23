import os
import sentencepiece as spm

# Caminho do dataset
data_file = "data/data.txt"

# Caminho onde o modelo será salvo
model_dir = "models"
model_prefix = os.path.join(model_dir, "m")

# Criar diretório caso não exista
os.makedirs(model_dir, exist_ok=True)

# Ajustar vocab_size dinamicamente (máximo de 33 conforme erro anterior)
vocab_size = min(33, 2000)

# Treinar o modelo SentencePiece
spm.SentencePieceTrainer.Train(
    f"--input={data_file} --model_prefix={model_prefix} --vocab_size={vocab_size}"
)

print(f"Treinamento concluído! O modelo foi salvo em '{model_dir}/'.")