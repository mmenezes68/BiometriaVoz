import os
import librosa
import numpy as np
import soundfile as sf  # Usaremos sf.read() para evitar audioread/aifc
from scipy.spatial.distance import cosine
from fpdf import FPDF

# 📌 Extrai características do áudio
def extract_features(audio_path, sr=22050):
    """ Extrai características MFCC do áudio para comparação """
    # Lendo o áudio usando soundfile
    audio, sr = sf.read(audio_path)

    # Se o áudio tiver mais de um canal, converte para mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Extraindo MFCCs (Mel-frequency cepstral coefficients)
    return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).flatten()

# 📌 Compara duas vozes
def compare_voices(audio1_path, audio2_path):
    """ Compara dois arquivos de áudio e retorna a similaridade """
    features1 = extract_features(audio1_path)
    features2 = extract_features(audio2_path)

    # Cálculo da distância do cosseno entre os vetores MFCCs
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
    print(f"Relatório salvo em {output_path}")

# 📌 Teste do script
if __name__ == "__main__":
    audio1_path = "data/raw/audio1.wav"
    audio2_path = "data/raw/audio2.wav"

    if not os.path.exists(audio1_path) or not os.path.exists(audio2_path):
        print("Erro: Arquivos de áudio não encontrados.")
    else:
        similarity = compare_voices(audio1_path, audio2_path)
        print(f"Similaridade entre as vozes: {similarity:.2f}")
        generate_report(audio1_path, audio2_path, similarity)