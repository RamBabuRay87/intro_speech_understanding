import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------
# Feature Extraction
# ---------------------------------------

def get_features(waveform, Fs):

    # pre-emphasis
    waveform = np.append(waveform[0], waveform[1:] - 0.97 * waveform[:-1])

    frame_len = int(0.004 * Fs)   # 4 ms
    step = int(0.002 * Fs)        # 2 ms

    frames = []
    for i in range(0, len(waveform) - frame_len, step):
        frame = waveform[i:i+frame_len]
        fft = np.abs(np.fft.fft(frame))
        fft = fft[:len(fft)//2]   # low frequency half
        frames.append(np.log(fft + 1e-6))

    features = np.array(frames)

    # -------- VAD --------
    vad_len = int(0.025 * Fs)
    vad_step = int(0.010 * Fs)

    energy = []
    for i in range(0, len(waveform) - vad_len, vad_step):
        frame = waveform[i:i+vad_len]
        energy.append(np.sum(frame**2))

    energy = np.array(energy)
    threshold = 0.1 * np.max(energy)

    labels = np.zeros(len(features), dtype=int)
    label = 1

    for i, e in enumerate(energy):
        if e > threshold:
            idx = i * vad_step // step
            for k in range(5):
                if idx + k < len(labels):
                    labels[idx + k] = label
            label += 1

    return features, labels


# ---------------------------------------
# Train Neural Network
# ---------------------------------------

def train_neuralnet(features, labels, iterations):

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    nfeats = features.shape[1]
    nlabels = 1 + int(labels.max())

    model = nn.Sequential(
        nn.LayerNorm(nfeats),
        nn.Linear(nfeats, nlabels)
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = []

    for i in range(iterations):
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return model, np.array(losses)


# ---------------------------------------
# Test Neural Network
# ---------------------------------------

def test_neuralnet(model, features):

    X = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        out = model(X)
        probs = F.softmax(out, dim=1)

    return probs.numpy()

