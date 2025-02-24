import sounddevice as sd
import numpy as np
import wave

# Configurações de áudio
SAMPLE_RATE = 44100  # Taxa de amostragem
DURATION = 5  # Duração da gravação em segundos
OUTPUT_FILE = "audio.wav"  # Nome do arquivo de saída

def record_audio():
    print("Gravando... Fale agora!")
    audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype=np.int16)
    sd.wait()  # Aguarda a gravação terminar
    print("Gravação concluída.")

    # Salvar o áudio gravado
    with wave.open(OUTPUT_FILE, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes (16 bits)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())

    print(f"Arquivo de áudio salvo como {OUTPUT_FILE}")

if __name__ == "__main__":
    record_audio()