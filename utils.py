import numpy as np
import librosa

def preprocess_audio(audio_file, max_len=174):
    """
    Convert uploaded audio file into model-ready features (MFCCs).
    """
    # Load audio
    y, sr = librosa.load(audio_file, sr=None)
    
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    
    # Pad / truncate to fixed length
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    
    return mfcc.T  # shape (time_steps, n_mfcc)