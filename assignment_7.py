import numpy as np

def major_chord(f, Fs):
    '''
    Generate a one-half-second major chord
    '''
    N = int(0.5 * Fs)          # half second
    n = np.arange(N)

    # Frequencies of the chord
    f_root = f
    f_third = f * (2 ** (4 / 12))   # major third
    f_fifth = f * (2 ** (7 / 12))   # major fifth

    # Generate chord
    x = (
        np.cos(2 * np.pi * f_root * n / Fs) +
        np.cos(2 * np.pi * f_third * n / Fs) +
        np.cos(2 * np.pi * f_fifth * n / Fs)
    )

    return x


def dft_matrix(N):
    '''
    Create a DFT transform matrix W of size N
    '''
    W = np.zeros((N, N), dtype=complex)

    for k in range(N):
        for n in range(N):
            W[k, n] = np.cos(2 * np.pi * k * n / N) - 1j * np.sin(2 * np.pi * k * n / N)

    return W


def spectral_analysis(x, Fs):
    '''
    Find the three loudest frequencies
    '''
    N = len(x)

    W = dft_matrix(N)
    X = W @ x                 # DFT
    magnitude = np.abs(X)

    freqs = np.arange(N) * Fs / N

    # Use only positive frequencies
    half = N // 2
    magnitude = magnitude[:half]
    freqs = freqs[:half]

    # Indices of three largest peaks
    idx = np.argsort(magnitude)[-3:]
    loud_freqs = np.sort(freqs[idx])

    return loud_freqs[0], loud_freqs[1], loud_freqs[2]


#  only i used for check output
Fs = 8000      # sampling frequency
f = 440        # root frequency (A note)

x = major_chord(f, Fs)
f1, f2, f3 = spectral_analysis(x, Fs)

print("Three dominant frequencies (Hz):")
print(f1, f2, f3)
