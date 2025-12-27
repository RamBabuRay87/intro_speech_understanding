import numpy as np

def VAD(waveform, Fs):
    '''
    Extract segments with energy > 10% of maximum energy
    using 25ms frame length and 10ms frame step.
    '''
    frame_len = int(0.025 * Fs)   # 25 ms
    step = int(0.010 * Fs)        # 10 ms

    N = len(waveform)
    num_frames = 1 + (N - frame_len) // step

    energies = []
    frames = []

    for i in range(num_frames):
        start = i * step
        frame = waveform[start:start + frame_len]
        energy = np.sum(frame ** 2)

        energies.append(energy)
        frames.append(frame)

    energies = np.array(energies)
    threshold = 0.1 * np.max(energies)

    segments = []
    current_segment = []

    for i, energy in enumerate(energies):
        if energy > threshold:
            current_segment.extend(frames[i])
        else:
            if len(current_segment) > 0:
                segments.append(np.array(current_segment))
                current_segment = []

    if len(current_segment) > 0:
        segments.append(np.array(current_segment))

    return segments


def segments_to_models(segments, Fs):
    '''
    Create average log-spectrum model for each segment.
    '''
    models = []

    frame_len = int(0.004 * Fs)   # 4 ms
    step = int(0.002 * Fs)        # 2 ms

    for seg in segments:
        # Pre-emphasis
        pre_emph = np.append(seg[0], seg[1:] - 0.97 * seg[:-1])

        N = len(pre_emph)
        num_frames = 1 + (N - frame_len) // step

        spectra = []

        for i in range(num_frames):
            start = i * step
            frame = pre_emph[start:start + frame_len]

            fft_mag = np.abs(np.fft.fft(frame))
            half = fft_mag[:len(fft_mag)//2]   # low-frequency half
            spectra.append(half)

        spectra = np.array(spectra)
        avg_spectrum = np.mean(np.log(1e-6 + spectra), axis=0)

        models.append(avg_spectrum)

    return models


def recognize_speech(testspeech, Fs, models, labels):
    '''
    Recognize test speech using cosine similarity.
    '''
    test_segments = VAD(testspeech, Fs)
    test_models = segments_to_models(test_segments, Fs)

    Y = len(models)
    K = len(test_models)

    sims = np.zeros((Y, K))
    test_outputs = []

    for k, test_model in enumerate(test_models):
        for y, model in enumerate(models):
            num = np.dot(model, test_model)
            den = np.linalg.norm(model) * np.linalg.norm(test_model)
            sims[y, k] = num / den

        best_index = np.argmax(sims[:, k])
        test_outputs.append(labels[best_index])

    return sims, test_outputs
