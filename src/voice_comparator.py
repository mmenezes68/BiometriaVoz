import os
import torch
import torch.nn.functional as F
import soundfile as sf
import torchaudio.transforms as transforms
import matplotlib.pyplot as plt
from fpdf import FPDF
from PIL import Image
import numpy as np
from datetime import datetime

# Diretórios de saída
REPORT_DIR = "data/reports/"
LOGO_PATH = "data/assets/logo_govpro.png"

# 📌 Garante que o diretório de saída exista
os.makedirs(REPORT_DIR, exist_ok=True)

# 📌 Função para verificar se o logotipo é um PNG válido
def validate_logo():
    try:
        with Image.open(LOGO_PATH) as img:
            if img.format != "PNG":
                raise ValueError("O logotipo não está no formato PNG.")
    except Exception as e:
        print(f"❌ Erro com o logotipo: {e}. Removendo do relatório.")
        return False
    return True

# 📌 Gera um nome de arquivo único para o relatório
def generate_report_filename():
    now = datetime.now()
    date_part = now.strftime("%Y%m%d.%H%M%S")
    
    # Contar quantos relatórios já existem para a mesma data
    existing_reports = sorted([f for f in os.listdir(REPORT_DIR) if f.startswith(f"GOVPRO.{date_part}")])
    sequence_number = len(existing_reports) + 1  # Incrementa o número do próximo relatório
    
    # Formata com 7 dígitos (zeros à esquerda)
    filename = f"GOVPRO.{date_part}.{sequence_number:07d}.pdf"
    return os.path.join(REPORT_DIR, filename), filename  # Retorna caminho completo e nome do arquivo

# 📌 Função para extrair características MFCC
def extract_features(audio_path):
    """ Extrai características MFCC do áudio para comparação """
    try:
        waveform, sample_rate = sf.read(audio_path, dtype="float32")

        # Se for estéreo, converte para mono
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)

        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)

        # Aplica MFCC
        mfcc_transform = transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=13,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23}
        )
        mfcc = mfcc_transform(waveform)
        return mfcc.mean(dim=2).squeeze().numpy().astype("float32")

    except Exception as e:
        print(f"❌ Erro ao extrair características de {audio_path}: {e}")
        return None

# 📌 Função para comparar vozes
def compare_voices(audio1_path, audio2_path):
    """ Compara duas amostras de áudio usando similaridade do cosseno """
    features1 = extract_features(audio1_path)
    features2 = extract_features(audio2_path)

    if features1 is None or features2 is None:
        return None

    tensor1 = torch.tensor(features1, dtype=torch.float32)
    tensor2 = torch.tensor(features2, dtype=torch.float32)

    min_length = min(tensor1.shape[0], tensor2.shape[0])
    tensor1, tensor2 = tensor1[:min_length], tensor2[:min_length]

    tensor1_mean = torch.mean(tensor1, dim=0)
    tensor2_mean = torch.mean(tensor2, dim=0)

    similarity = F.cosine_similarity(tensor1_mean, tensor2_mean, dim=0).item()
    return similarity

# 📌 Gera gráficos de comparação das formas de onda
def generate_audio_plot(audio1_path, audio2_path, output_path):
    """ Gera gráficos das formas de onda dos áudios """
    waveform1, sr1 = sf.read(audio1_path)
    waveform2, sr2 = sf.read(audio2_path)

    # Se for estéreo, converte para mono
    if len(waveform1.shape) > 1:
        waveform1 = waveform1.mean(axis=1)
    if len(waveform2.shape) > 1:
        waveform2 = waveform2.mean(axis=1)

    plt.figure(figsize=(10, 4))
    plt.subplot(2, 1, 1)
    plt.plot(waveform1, label="Áudio 1", color="blue")
    plt.title("Forma de Onda - Áudio 1")
    plt.xlabel("Tempo")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(waveform2, label="Áudio 2", color="red")
    plt.title("Forma de Onda - Áudio 2")
    plt.xlabel("Tempo")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# 📌 Gera o relatório em PDF
def generate_pdf_report(audio1_path, audio2_path, similarity_score, client_name="Não Informado", microphone="Não Informado"):
    """ Gera um relatório em PDF com os resultados da comparação de vozes """
    try:
        output_path, filename = generate_report_filename()
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Logotipo
        if validate_logo():
            pdf.image(LOGO_PATH, x=10, y=10, w=50)

        # Nome do relatório
        pdf.set_font("Arial", style='B', size=10)
        pdf.cell(200, 10, f"Relatório: {filename}", ln=True, align="C")
        pdf.ln(5)

        # Cabeçalho
        pdf.set_font("Arial", style='B', size=16)
        pdf.cell(200, 10, "Relatório de Comparação de Voz", ln=True, align="C")
        pdf.ln(15)

        # Informações sobre o teste
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, f"Cliente: {client_name}", ln=True)
        pdf.cell(200, 10, f"Microfone Utilizado: {microphone}", ln=True)
        pdf.cell(200, 10, f"Arquivo 1: {os.path.basename(audio1_path)}", ln=True)
        pdf.cell(200, 10, f"Arquivo 2: {os.path.basename(audio2_path)}", ln=True)
        pdf.ln(10)

        # Resultado da comparação
        pdf.set_font("Arial", style='B', size=14)
        pdf.cell(200, 10, f"Similaridade entre as vozes: {similarity_score:.2f}", ln=True)
        pdf.ln(10)

        # Método utilizado
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 8, (
            "A comparação foi feita utilizando coeficientes MFCC (Mel-Frequency Cepstral Coefficients) "
            "e a similaridade do cosseno. Valores próximos de 1 indicam alta similaridade entre as vozes, "
            "enquanto valores próximos de 0 indicam baixa similaridade."
        ))
        pdf.ln(15)

        # Gerar gráfico e adicioná-lo ao relatório
        plot_path = "data/assets/audio_comparison.png"
        generate_audio_plot(audio1_path, audio2_path, plot_path)
        pdf.image(plot_path, x=10, w=180)

        # Salvar PDF
        pdf.output(output_path)
        print(f"📄 Relatório salvo como {output_path}")

    except Exception as e:
        print(f"❌ Erro ao gerar o relatório em PDF: {e}")