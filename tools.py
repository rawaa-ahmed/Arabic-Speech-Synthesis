import json
import os

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy.io import wavfile
from hifigan import AttrDict, Generator
matplotlib.use("Agg")
import constants

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(data, device):
    if len(data) == 12:
        (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
        ) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        mels = torch.from_numpy(mels).float().to(device)
        mel_lens = torch.from_numpy(mel_lens).to(device)
        pitches = torch.from_numpy(pitches).float().to(device)
        energies = torch.from_numpy(energies).to(device)
        durations = torch.from_numpy(durations).long().to(device)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
        )

    if len(data) == 6:
        (ids, raw_texts, speakers, texts, src_lens, max_src_len) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)

        return (ids, raw_texts, speakers, texts, src_lens, max_src_len)


def log(logger, step=None, losses=None, fig=None, audio=None, sampling_rate=22050, tag=""):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/mel_loss", losses[1], step)
        logger.add_scalar("Loss/mel_postnet_loss", losses[2], step)
        logger.add_scalar("Loss/pitch_loss", losses[3], step)
        logger.add_scalar("Loss/energy_loss", losses[4], step)
        logger.add_scalar("Loss/duration_loss", losses[5], step)

    if fig is not None:
        logger.add_figure(tag, fig)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
        )


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(
        0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def expand(values, durations):
    out = []
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def synth_one_sample(targets, predictions, vocoder):

    basename = targets[0][0]
    src_len = predictions[7][0].item()
    mel_len = predictions[8][0].item()
    mel_target = targets[5][0, :mel_len].detach().transpose(0, 1)
    mel_prediction = predictions[1][0, :mel_len].detach().transpose(0, 1)
    duration = targets[10][0, :src_len].detach().cpu().numpy()

    pitch = targets[8][0, :src_len].detach().cpu().numpy()
    pitch = expand(pitch, duration)

    energy = targets[9][0, :src_len].detach().cpu().numpy()
    energy = expand(energy, duration)

    with open(constants.DATA_PATH + "/stats.json") as f:
        stats = json.load(f)
        stats = stats["pitch"] + stats["energy"][:2]
    fig = plot_mel(
        [
            (mel_prediction.cpu().numpy(), pitch, energy),
            (mel_target.cpu().numpy(), pitch, energy),
        ],
        stats,
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram"],
    )

    if vocoder is not None:
        wav_reconstruction = vocoder_infer(
            mel_target.unsqueeze(0),
            vocoder,
            constants.MAX_WAVE_VALUE
        )[0]
        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            constants.MAX_WAVE_VALUE
        )[0]
    else:
        wav_reconstruction = wav_prediction = None

    return fig, wav_reconstruction, wav_prediction, basename


def synth_samples(targets, predictions, vocoder, save_path):

    basenames = targets[0]
    for i in range(len(predictions[0])):
        # basename = targets[0][i]
        basename = basenames[i]
        src_len = predictions[8][i].item()
        mel_len = predictions[9][i].item()
        mel_prediction = predictions[1][i, :mel_len].detach().transpose(0, 1)
        duration = predictions[5][i, :src_len].detach().cpu().numpy()
        pitch = predictions[2][i, :src_len].detach().cpu().numpy()
        pitch = expand(pitch, duration)
        energy = predictions[3][i, :src_len].detach().cpu().numpy()
        energy = expand(energy, duration)

        with open(constants.DATA_PATH + "/stats.json") as f:
            stats = json.load(f)
            stats = stats["pitch"] + stats["energy"][:2]

            fig = plot_mel(
                [
                    (mel_prediction.cpu().numpy(), pitch, energy),
                ],
                stats,
                ["Synthetized Spectrogram", "Ground-Truth Spectrogram"],
            )
            plt.savefig(os.path.join(constants.RESULT_PATH,
                        "{}.png".format(basename)))
            plt.close()

    mel_predictions = predictions[1].transpose(1, 2)
    lengths = predictions[9] * constants.STFT_HOP_LENGTH
    print('vocoder inference...')
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, constants.MAX_WAVE_VALUE, lengths=lengths)
    print('saving waveform to ', save_path)
    for wav, basename in zip(wav_predictions, basenames):
        wavfile.write(save_path, constants.SAMPLING_RATE, wav)


def plot_mel(data, stats, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats
    pitch_min = pitch_min * pitch_std + pitch_mean
    pitch_max = pitch_max * pitch_std + pitch_mean

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        mel, pitch, energy = data[i]
        pitch = pitch * pitch_std + pitch_mean
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small",
                               left=False, labelleft=False)
        axes[i][0].set_anchor("W")

        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(pitch, color="tomato")
        ax1.set_xlim(0, mel.shape[1])
        ax1.set_ylim(0, pitch_max)
        ax1.set_ylabel("F0", color="tomato")
        ax1.tick_params(labelsize="x-small", colors="tomato",
                        bottom=False, labelbottom=False)

        ax2 = add_axis(fig, axes[i][0])
        ax2.plot(energy, color="darkviolet")
        ax2.set_xlim(0, mel.shape[1])
        ax2.set_ylim(energy_min, energy_max)
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(
            labelsize="x-small",
            colors="darkviolet",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )

    return fig


def pad_1D(inputs, PAD=0):

    max_len = max((len(x) for x in inputs))
    inputs_padded = []
    for x in inputs:
        x_padded = np.pad(
            x, (0, max_len - x.shape[0]), mode="constant", constant_values=PAD)
        inputs_padded.append(x_padded)

    padded = np.stack(inputs_padded, 0)

    return padded


def pad_2D(inputs, maxlen=None):
    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
    inputs_padded = []
    for x in inputs:

        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(x, (0, max_len - np.shape(x)
                          [0]), mode="constant", constant_values=0)
        inputs_padded.append(x_padded)

    output = np.stack(inputs_padded)

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])
    out_list = []
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0)
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0)
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


def get_vocoder(device, vocoder_config_path=None, speaker_pre_trained_path=None):
    with open(vocoder_config_path, "r") as f:
        config = json.load(f)
        config = AttrDict(config)
        vocoder = Generator(config)
        ckpt = torch.load(speaker_pre_trained_path,
                          map_location=torch.device("cpu"))
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, max_wav_value, lengths=None):
    with torch.no_grad():
        wavs = vocoder(mels).squeeze(1)
    wavs = (wavs.cpu().numpy() * max_wav_value).astype("int16")
    wavs = [wav for wav in wavs]
    if lengths is not None:
        m = len(mels)
        for i in range(m):
            wavs[i] = wavs[i][: lengths[i]]
    return wavs