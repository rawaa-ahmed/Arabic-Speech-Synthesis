import numpy as np
import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn
from librosa.util import pad_center  # , tiny, normalize
from scipy.signal import get_window


# # An implementaion of librosa window sumsquare window envelope function
# def window_sumsquare(
#     window,
#     n_frames,
#     hop_length=512,
#     win_length=None,
#     n_fft=2048,
#     dtype=np.float32,
#     norm=None,
# ):
#     """Compute the sum-square envelope of a window function at a given hop length.

#     This is used to estimate modulation effects induced by windowing observations
#     in short-time Fourier transforms.

#     Parameters
#     ----------
#     window : string, tuple, number, callable, or list-like
#         Window specification, as in `get_window`
#     n_frames : int > 0
#         The number of analysis frames
#     hop_length : int > 0
#         The number of samples to advance between frames
#     win_length : [optional]
#         The length of the window function.  By default, this matches ``n_fft``.
#     n_fft : int > 0
#         The length of each analysis frame.
#     dtype : np.dtype
#         The data type of the output
#     norm : {np.inf, -np.inf, 0, float > 0, None}
#         Normalization mode used in window construction.
#         Note that this does not affect the squaring operation.

#     Returns
#     -------
#     wss : np.ndarray, shape=``(n_fft + hop_length * (n_frames - 1))``
#         The sum-squared envelope of the window function
#     """
#     if win_length is None:
#         win_length = n_fft
#     n = n_fft + hop_length * (n_frames - 1)
#     x = np.zeros(n, dtype=dtype)
#     # Compute the squared window at the desired length
#     win = get_window(window, win_length)
#     win_sq = normalize(win, norm=norm) ** 2
#     win_sq = pad_center(win_sq, n_fft)
#     # Fill the envelope
#     for i in range(n_frames):
#         sample = i * hop_length
#         l=max(0, min(n_fft, n - sample))
#         k=min(n, sample + n_fft)
#         x[sample : k] += win_sq[: l]
#     return x

def DFT(x):
    '''
    Discrete Fourier Transform of a 1-D array, x.
    '''
    x = np.asarray(x, dtype=np.float32)
    N = x.shape[0]
    n = np.arange(N)
    # the value of k is supplied by IDFT. For every value of k, entire n will be used. Hence the reshape.
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n/N)

    return M@x


def FFT(x):
    '''
    Fast Fourier Transform of a 1-D array
    '''
    x = np.asarray(x, dtype=np.float32)
    N = x.shape[0]

    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2!")
    elif N <= 32:  # base case
        return DFT(x)
    else:
        X_even = FFT(x[::2])  # choose every 2nd data from 0 (even numbered)
        X_odd = FFT(x[1::2])  # choose every 2nd data from 1 (odd numbered)
        factor = np.exp(-2j * np.pi * np.arange(N) / N)

        return np.concatenate([X_even + factor[:(N // 2)] * X_odd,
                               X_even + factor[(N // 2):] * X_odd])


class TacotronSTFT(torch.nn.Module):
    def __init__(
        self,
        filter_length,
        hop_length,
        win_length,
        n_mel_channels,
        sampling_rate,
        mel_fmin,
        mel_fmax,
        window="hann"
    ):
        super(TacotronSTFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        mel_basis = librosa_mel_fn(
            sr=sampling_rate, n_fft=filter_length, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        scale = self.filter_length / self.hop_length
        fourier_basis = FFT(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])])

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        # inverse_basis = torch.FloatTensor(np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        assert filter_length >= win_length
        # get window and zero center pad it to filter_length
        fft_window = get_window(window, win_length, fftbins=True)
        fft_window = pad_center(data=fft_window, size=filter_length)
        fft_window = torch.from_numpy(fft_window).float()

        # window the bases
        forward_basis *= fft_window
        # inverse_basis *= fft_window

        self.register_buffer("forward_basis", forward_basis.float())
        # self.register_buffer("inverse_basis", inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode="reflect",
        )
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data.cuda(),
            torch.autograd.Variable(
                self.forward_basis, requires_grad=False).cuda(),
            stride=self.hop_length,
            padding=0,
        ).cpu()

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.autograd.Variable(
            torch.atan2(imag_part.data, real_part.data))

        return (magnitude, phase)

    # def inverse(self, magnitude, phase):
    #     recombine_magnitude_phase = torch.cat([magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1)

    #     inverse_transform = F.conv_transpose1d(
    #         recombine_magnitude_phase,
    #         torch.autograd.Variable(self.inverse_basis, requires_grad=False),
    #         stride=self.hop_length,
    #         padding=0,
    #     )

    #     if self.window is not None:
    #         window_sum = window_sumsquare(
    #             self.window,
    #             magnitude.size(-1),
    #             hop_length=self.hop_length,
    #             win_length=self.win_length,
    #             n_fft=self.filter_length,
    #             dtype=np.float32,
    #         )
    #         # remove modulation effects
    #         approx_nonzero_indices = torch.from_numpy(np.where(window_sum > tiny(window_sum))[0])
    #         window_sum = torch.autograd.Variable(torch.from_numpy(window_sum), requires_grad=False)
    #         window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum
    #         inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

    #       STFT hop ratio
    #         inverse_transform *= float(self.filter_length) / self.hop_length

    #     inverse_transform = inverse_transform[:, :, int(self.filter_length / 2) :]
    #     inverse_transform = inverse_transform[:, :, : -int(self.filter_length / 2) :]

    #     return inverse_transform

    # def forward(self, input_data):
    #     self.magnitude, self.phase = self.transform(input_data)
    #     reconstruction = self.inverse(self.magnitude, self.phase)
    #     return reconstruction
    def spectral_normalize(self, magnitudes, C=1, clip_val=1e-5):
        return torch.log(torch.clamp(magnitudes, min=clip_val) * C)

    def spectral_de_normalize(self, magnitudes, C=1):
        return torch.exp(magnitudes) / C

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert torch.min(y.data) >= -1
        assert torch.max(y.data) <= 1
        magnitudes, _ = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = self.spectral_normalize(
            torch.matmul(self.mel_basis, magnitudes))
        energy = torch.norm(magnitudes, dim=1)

        return mel_output, energy
