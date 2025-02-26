import os
import torch
import torchaudio
import torchaudio.transforms as transforms
import numpy as np
from scipy.spatial.distance import cosine
from fpdf import FPDF

# 📌 Extrai características do áudio usando torchaudio
def extract_features(audio_path):
    """ Extrai características MFCC do áudio para comparação """
    waveform, sample_rate = torchaudio.load(audio_path)

    # Converte para mono se houver mais de um canal
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Aplica a transformação MFCC
    mfcc_transform = transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=13,  # Número de coeficientes MFCC
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23}
    )
    mfcc = mfcc_transform(waveform)

    return mfcc.mean(dim=2).squeeze().numpy()  # Retorna os coeficientes MFCC

# 📌 Compara duas vozes usando distância do cosseno
def compare_voices(audio1_path, audio2_path):
    """ Compara dois arquivos de áudio e retorna a similaridade """
    features1 = extract_features(audio1_path)
    features2 = extract_features(audio2_path)

    # Verifica se os vetores têm o mesmo tamanho
    if features1.shape != features2.shape:
        min_length = min(features1.shape[0], features2.shape[0])
        features1 = features1[:min_length]
        features2 = features2[:min_length]

    # Calcula a similaridade do cosseno entre os vetores MFCCs
    similarity = 1 - cosine(features1, features2)
    return similarity

# 📌 Gera um relatório em PDF com os resultados da análise
def generate_report(audio1_path, audio2_path, similarity, output_path="report.pdf"):
    """ Gera um laudo em PDF com os resultados da comparação """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, "Relatório de Comparação de Voz", ln=True, align="C")
    pdf.ln(10)
    pdf.cell(200, 10, f"Arquivo 1: {os.path.basename(audio1_path)}", ln=True)
    pdf.cell(200, 10, f"Arquivo 2: {os.path.basename(audio2_path)}", ln=True)
    pdf.cell(200, 10, f"Similaridade: {similarity:.2f}", ln=True)

    pdf.output(output_path)
    print(f"📄 Relatório salvo em {output_path}")

# 📌 Teste do script
if __name__ == "__main__":
    audio1_path = "data/raw/audio1.wav"
    audio2_path = "data/raw/audio2.wav"

    if not os.path.exists(audio1_path) or not os.path.exists(audio2_path):
        print("❌ Erro: Arquivos de áudio não encontrados.")
    else:
        similarity = compare_voices(audio1_path, audio2_path)
        print(f"🆚 Similaridade entre as vozes: {similarity:.2f}")
        generate_report(audio1_path, audio2_path, similarity)