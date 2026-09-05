import numpy as np
import librosa

SR = 22050
DURATION = 4.0
N_MFCC = 40
MAX_SAMPLES = int(SR * DURATION)


def load_audio(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    if len(y) < MAX_SAMPLES:
        y = np.pad(y, (0, MAX_SAMPLES - len(y)))
    else:
        y = y[:MAX_SAMPLES]
    peak = np.max(np.abs(y)) if len(y) else 0
    if peak > 0:
        y = y / peak
    return y.astype(np.float32)


def extract_mfcc(path, augment=False):
    y = load_audio(path)
    if augment:
        choice = np.random.randint(0, 4)
        if choice == 1:
            y = y + 0.003 * np.random.randn(len(y)).astype(np.float32)
        elif choice == 2:
            y = librosa.effects.time_stretch(y, rate=np.random.uniform(0.92, 1.08))
            y = np.pad(y[:MAX_SAMPLES], (0, max(0, MAX_SAMPLES - len(y))))
        elif choice == 3:
            y = librosa.effects.pitch_shift(y, sr=SR, n_steps=np.random.uniform(-1.5, 1.5))

    mfcc = librosa.feature.mfcc(
        y=y, sr=SR, n_mfcc=N_MFCC, n_fft=2048, hop_length=512, n_mels=64
    )
    mean = mfcc.mean(axis=1, keepdims=True)
    std = mfcc.std(axis=1, keepdims=True) + 1e-6
    return ((mfcc - mean) / std).astype(np.float32)
