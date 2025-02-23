from transformers import AutoTokenizer, AutoModel
import torch

# Carregar o modelo e o tokenizador
modelo_nome = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(modelo_nome)
modelo = AutoModel.from_pretrained(modelo_nome)

# Exemplo de frase para teste
texto = "O aprendizado de máquina está revolucionando a inteligência artificial."

# Tokenizar o texto
tokens = tokenizer(texto, return_tensors="pt")

# Passar os tokens pelo modelo
with torch.no_grad():
    output = modelo(**tokens)

# Mostrar os tokens
print("Tokens:", tokenizer.convert_ids_to_tokens(tokens["input_ids"][0]))
print("Saída do modelo (embedding shape):", output.last_hidden_state.shape)