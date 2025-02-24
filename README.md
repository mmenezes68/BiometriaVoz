# Documentação do Sistema de Biometria de Voz

## 1. Introdução
O sistema de **Biometria de Voz** tem como objetivo comparar vozes gravadas para determinar se pertencem à mesma pessoa. Ele pode ser utilizado para fins forenses, controle de acesso e identificação de usuários com base em características vocais.

## 2. Requisitos
Para rodar o sistema corretamente, é necessário instalar os seguintes pacotes Python:

```sh
pip install librosa soundfile scipy numpy
```

Além disso, é recomendável rodar o sistema em um ambiente virtual (venv).

## 3. Estrutura do Projeto
Abaixo está a organização dos diretórios e arquivos principais:

```
BiometriaVoz/
│── Documentação/
│── Processamento/
│── data/
│   ├── data.txt
│   ├── processed/
│   ├── raw/
│       └── audio.wav
│── models/
│── notebooks/
│── src/
│   ├── capture_audio.py
│   ├── extract_features.py
│   ├── process_audio.py
│   ├── train_biometria.py
│   └── voice_comparator.py
│── tests/
│── README.md
```

## 4. Fluxo de Funcionamento
1. **Captura de Áudio** – O sistema grava um arquivo de áudio ou recebe um arquivo já existente.
2. **Extração de Características** – O áudio é processado para extrair padrões únicos da voz.
3. **Comparação** – O sistema compara duas ou mais vozes para determinar a similaridade.
4. **Geração de Relatório** – Um laudo é gerado com os resultados da análise.

## 5. Uso
Para capturar áudio:
```sh
python src/capture_audio.py
```

Para comparar duas vozes:
```sh
python src/voice_comparator.py
```

## 6. Saída e Relatório
O sistema gera um **arquivo de texto (TXT ou PDF)** detalhando a comparação, incluindo:
- Arquivos analisados
- Métodos usados
- Nível de similaridade entre as vozes
- Data e hora da análise

A estrutura do relatório será expandida conforme necessidade.

---

*Este documento pode ser atualizado à medida que o projeto evolui.*