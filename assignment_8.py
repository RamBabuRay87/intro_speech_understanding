import numpy as np

def waveform_to_frames(waveform, frame_length, step):
    '''
    Chop a waveform into overlapping frames.
    '''
    N = len(waveform)
    num_frames = 1 + (N - frame_length) // step

    frames = np.zeros((num_frames, frame_length))

    for i in range(num_frames):
        start = i * step
        frames[i, :] = waveform[start:start + frame_length]

    return frames


def frames_to_mstft(frames):
    '''
    Take the magnitude FFT of every row of the frames matrix.
    '''
    # FFT along each frame (row)
    fft_frames = np.fft.fft(frames, axis=1)

    # Magnitude STFT
    mstft = np.abs(fft_frames)

    return mstft


def mstft_to_spectrogram(mstft):
    '''
    Convert magnitude STFT to spectrogram in decibels.
    '''
    # Avoid log of zero
    floor = 0.001 * np.amax(mstft)
    mstft_safe = np.maximum(mstft, floor)

    # Convert to decibels
    spectrogram = 20 * np.log10(mstft_safe)

    # Limit dynamic range to 60 dB
    max_val = np.amax(spectrogram)
    spectrogram = np.maximum(spectrogram, max_val - 60)

    return spectrogram


# Example waveform, just check output
waveform = np.random.randn(16000)

frames = waveform_to_frames(waveform, frame_length=400, step=160)
mstft = frames_to_mstft(frames)
spec = mstft_to_spectrogram(mstft)

print(frames.shape)
print(mstft.shape)
print(spec.shape)
