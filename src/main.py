import os
import datetime
import torch
import torch.nn.functional as F
import soundfile as sf
import torchaudio.transforms as transforms
import matplotlib.pyplot as plt
from fpdf import FPDF
from PIL import Image
import numpy as np

# Caminhos fixos
LOGO_PATH = "data/assets/logo_govpro.png"
REPORTS_DIR = "data/reports/"  # Diretório onde os relatórios serão armazenados

# 📌 Garante que o diretório de relatórios exista
os.makedirs(REPORTS_DIR, exist_ok=True)

# 📌 Função para gerar um nome de arquivo único e sequencial para o relatório
def generate_report_filename():
    """Gera um nome de relatório no formato GOVPRO_YYYYMMDD_HHMMSS_000000X.pdf"""
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # Listar relatórios já existentes no diretório
    existing_reports = [f for f in os.listdir(REPORTS_DIR) if f.startswith("GOVPRO_")]

    if existing_reports:
        # Extrai os números sequenciais dos arquivos existentes
        existing_numbers = [int(f.split("_")[-1].split(".")[0]) for f in existing_reports if f.split("_")[-1].split(".")[0].isdigit()]
        max_number = max(existing_numbers) + 1 if existing_numbers else 1
    else:
        max_number = 1

    # Formata o número sequencial com 7 dígitos
    report_filename = os.path.join(REPORTS_DIR, f"GOVPRO_{timestamp}_{max_number:07d}.pdf")

    return report_filename

# 📌 Função para verificar se o logotipo é válido
def validate_logo():
    try:
        with Image.open(LOGO_PATH) as img:
            if img.format != "PNG":
                raise ValueError("O logotipo não está no formato PNG.")
    except Exception as e:
        print(f"❌ Erro com o logotipo: {e}. Removendo do relatório.")
        return False
    return True

# 📌 Função para extrair características MFCC
def extract_features(audio_path):
    try:
        waveform, sample_rate = sf.read(audio_path, dtype="float32")
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)
        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        mfcc_transform = transforms.MFCC(sample_rate=sample_rate, n_mfcc=13,
                                         melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23})
        mfcc = mfcc_transform(waveform)
        return mfcc.mean(dim=2).squeeze().numpy().astype("float32")
    except Exception as e:
        print(f"❌ Erro ao extrair características de {audio_path}: {e}")
        return None

# 📌 Função para comparar vozes
def compare_voices(audio1_path, audio2_path):
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

# 📌 Função para gerar gráfico das formas de onda
def generate_audio_plot(audio1_path, audio2_path, output_path):
    waveform1, sr1 = sf.read(audio1_path)
    waveform2, sr2 = sf.read(audio2_path)
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

# 📌 Função para gerar o relatório em PDF
def generate_pdf_report(audio1_path, audio2_path, similarity_score, client_name="Não Informado", microphone="Não Informado"):
    report_filename = generate_report_filename()

    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Cabeçalho com logo e título separados corretamente
        if validate_logo():
            pdf.image(LOGO_PATH, x=10, y=10, w=50)
        pdf.set_font("Arial", style='B', size=16)
        pdf.cell(190, 10, "Relatório de Comparação de Voz", ln=True, align="C")
        
        # Número do relatório igual ao nome do arquivo gerado (sem .pdf)
        report_number = os.path.basename(report_filename).replace(".pdf", "")
        pdf.set_font("Arial", size=12)
        pdf.cell(190, 10, f"Número do Relatório: {report_number}", ln=True, align="C")
        pdf.ln(10)

        # Informações do cliente
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, f"Cliente: {client_name}", ln=True)
        if microphone != "Não Informado":
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
        pdf.multi_cell(0, 8, ("A comparação foi realizada utilizando coeficientes MFCC "
                              "(Mel-Frequency Cepstral Coefficients) e a similaridade do cosseno. "
                              "Valores próximos de 1 indicam alta similaridade entre as vozes, "
                              "enquanto valores próximos de 0 indicam baixa similaridade."))
        pdf.ln(10)

        # Gera o gráfico e adiciona ao PDF
        plot_path = os.path.join("data/assets", "audio_comparison.png")
        generate_audio_plot(audio1_path, audio2_path, plot_path)
        pdf.image(plot_path, x=10, w=180)

        # Salvar PDF
        pdf.output(report_filename)
        print(f"📄 Relatório salvo como {report_filename}")
    except Exception as e:
        print(f"❌ Erro ao gerar o relatório em PDF: {e}")

# 📌 Execução principal do programa
if __name__ == "__main__":
    audio1_path = "data/raw/audio1.wav"
    audio2_path = "data/raw/audio2.wav"
    client_name = input("Digite o nome do cliente: ") or "Não Informado"
    microphone = input("Digite o modelo do microfone utilizado: ") or "Não Informado"
    similarity = compare_voices(audio1_path, audio2_path)
    if similarity is not None:
        generate_pdf_report(audio1_path, audio2_path, similarity, client_name, microphone)
    else:
        print("❌ Erro ao calcular similaridade.")