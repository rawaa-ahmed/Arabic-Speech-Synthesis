import numpy as np
import torch
from scipy.io.wavfile import write

"""
The Griffin-Lim Algorithm (GLA) is a phase reconstruction method based on the redundancy of the short-time Fourier transform. 
It promotes the consistency of a spectrogram by iterating two projections, 
where a spectrogram is said to be consistent when its inter-bin dependency owing to the redundancy of STFT is retained. 
GLA is based only on the consistency and does not take any prior knowledge about the target signal into account.
"""


def griffin_lim(magnitudes, stft_fn, n_iters=30):
    """
    PARAMS
    ------
    magnitudes: spectrogram magnitudes
    stft_fn: STFT class with transform (STFT) and inverse (ISTFT) methods
    """

    angles = np.angle(np.exp(2j * np.pi * np.random.rand(*magnitudes.size())))
    angles = angles.astype(np.float32)
    angles = torch.autograd.Variable(torch.from_numpy(angles))
    signal = stft_fn.inverse(magnitudes, angles).squeeze(1)

    for i in range(n_iters):
        _, angles = stft_fn.transform(signal)
        signal = stft_fn.inverse(magnitudes, angles).squeeze(1)
    return signal


def get_mel_from_wav(audio, _stft):
    audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
    audio = torch.autograd.Variable(audio, requires_grad=False)
    melspec, energy = _stft.mel_spectrogram(audio)
    melspec = melspec.squeeze(0).numpy().astype(np.float32)
    energy = energy.squeeze(0).numpy().astype(np.float32)
    return melspec, energy


def inv_mel_spec(mel, out_filename, _stft, griffin_iters=60):
    mel_decompress = _stft.spectral_de_normalize(
        torch.stack([mel])).transpose(1, 2).data.cpu()
    spec_from_mel = torch.mm(
        mel_decompress[0], _stft.mel_basis).transpose(0, 1)
    # multiply by scale=1000
    spec_from_mel = spec_from_mel.unsqueeze(0) * 1000
    audio = torch.autograd.Variable(spec_from_mel[:, :, :-1])
    audio = griffin_lim(audio, _stft.stft_fn, griffin_iters)
    audio = audio.squeeze().cpu().numpy()
    write(out_filename, _stft.sampling_rate, audio)
