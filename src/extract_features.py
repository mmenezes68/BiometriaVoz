import torchaudio
import torchaudio.transforms as transforms

def extract_features(audio_file):
    """
    Extrai características MFCC do áudio para comparação usando torchaudio.
    """
    try:
        # Carrega o arquivo de áudio
        waveform, sample_rate = torchaudio.load(audio_file)

        # Aplica a transformação MFCC
        mfcc_transform = transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=13,  # Número de coeficientes MFCC
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23}
        )
        mfcc = mfcc_transform(waveform)

        # Retorna os coeficientes MFCC como um array numpy
        return mfcc.mean(dim=2).squeeze().numpy()

    except Exception as e:
        print(f"❌ Erro ao extrair características de {audio_file}: {e}")
        return None