import os
import sys

# Garante que o diretório raiz do projeto esteja no sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.extract_features as extract
import src.voice_comparator as comparator

def main():
    print("\n🔹 Sistema de Biometria de Voz 🔹\n")
    
    # Definir caminhos dos arquivos de áudio
    audio_file1 = "data/raw/audio1.wav"
    audio_file2 = "data/raw/audio2.wav"
    
    # Verificar se os arquivos existem
    missing_files = [f for f in [audio_file1, audio_file2] if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Arquivos de áudio não encontrados: {', '.join(missing_files)}")
        print("🔎 Verifique o diretório `data/raw/` e tente novamente.")
        sys.exit(1)

    # Extração de características usando torchaudio
    print("🎵 Extraindo características dos áudios...")
    features1 = extract.extract_features(audio_file1)
    features2 = extract.extract_features(audio_file2)

    if features1 is None or features2 is None:
        print("❌ Erro na extração de características. Verifique os arquivos de áudio.")
        sys.exit(1)

    # Comparação de vozes
    print("🆚 Comparando vozes...")
    similarity_score = comparator.compare_voices(features1, features2)

    # Geração do relatório em PDF
    print("📄 Gerando relatório...")
    comparator.generate_pdf_report(audio_file1, audio_file2, similarity_score)
    
    print("\n✅ Comparação concluída. Relatório salvo em `report.pdf`\n")

if __name__ == "__main__":
    main()