import os
import src.extract_features as extract
import src.voice_comparator as comparator
import src.plot_utils as plotter
import src.generate_pdf as pdf_generator

def main():
    # Definir os caminhos dos arquivos de áudio
    audio1_path = "data/raw/audio1.wav"
    audio2_path = "data/raw/audio2.wav"

    # Extrair características
    features1 = extract.extract_features(audio1_path)
    features2 = extract.extract_features(audio2_path)

    # Comparar vozes
    similarity_score = comparator.compare_voices(features1, features2)
    print(f"Similaridade entre as vozes: {similarity_score:.2f}")

    # Gerar gráfico de similaridade
    chart_path = "reports/similarity_chart.png"
    plotter.generate_similarity_chart(similarity_score, chart_path)

    # Criar relatório em PDF
    report_path = "reports/voice_comparison_report.pdf"
    pdf_report = pdf_generator.PDFReport()
    pdf_report.create_report(audio1_path, audio2_path, similarity_score, chart_path, report_path)

    print(f"Relatório salvo em: {report_path}")

if __name__ == "__main__":
    main()