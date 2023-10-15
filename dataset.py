import json
import os
import numpy as np
from torch.utils.data import Dataset
from text import text_to_sequence
from tools import pad_1D, pad_2D
import constants

# read data from training.txt or validation.txt


def get_data(file_path):
    names = []
    texts = []
    raw_text = []
    speakers = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            g = line.strip("\n").split("|")
            names.append(g[0])
            speakers.append(g[1])
            texts.append(g[2])
            raw_text.append(g[3])
    return names, speakers, texts, raw_text

# For preprocessed data: mels, pitch, durations, and energy


class Dataset(Dataset):
    def __init__(self, file_name, sort=False, drop_last=False):
        # self.dataset_name = constants.DATA_PATH
        self.data_path = constants.DATA_PATH
        self.batch_size = constants.BATCH_SIZE
        self.basename, self.speaker, self.text, self.raw_text = self.get_data(
            file_name)
        with open(os.path.join(constants.DATA_PATH, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx]))
        mel = np.load(
            self.data_path+'/mel/{}-mel-{}.npy'.format(speaker, basename), allow_pickle=True)
        pitch = np.load(
            self.data_path+'/pitch/{}-pitch-{}.npy'.format(speaker, basename), allow_pickle=True)
        energy = np.load(
            self.data_path+'/energy/{}-energy-{}.npy'.format(speaker, basename), allow_pickle=True)
        duration = np.load(
            self.data_path + '/duration/{}-duration-{}.npy'.format(speaker, basename), allow_pickle=True)

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
        }

        return sample

    def get_data(self, file_name):
        return get_data(file_path=self.data_path + '/' + file_name)

    def reprocess(self, data, idxs):
        ids = []
        speakers = []
        texts = []
        raw_texts = []
        mels = []
        pitches = []
        energies = []
        durations = []
        text_lens = []
        mel_lens = []
        for idx in idxs:
            ids.append(data[idx]["id"])
            speakers.append(data[idx]["speaker"])
            texts.append(data[idx]["text"])
            raw_texts.append(data[idx]["raw_text"])
            mels.append(data[idx]["mel"])
            pitches.append(data[idx]["pitch"])
            energies.append(data[idx]["energy"])
            durations.append(data[idx]["duration"])
            text_lens.append(texts[-1].shape[0])
            mel_lens.append(mels[-1].shape[0])

        text_lens = np.array(text_lens)
        mel_lens = np.array(mel_lens)
        speakers = np.array(speakers)

        # Applying padding so all elements of each of them are of the same length
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
            pitches,
            energies,
            durations,
        )

    def collate_fn(self, data):
        data_size = len(data)
        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)
        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size):]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]
        output = []
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))
        return output

# For text data in training.txt and validation.txt


class TextDataset(Dataset):
    def __init__(self, file_path):

        self.basename, self.speaker, self.text, self.raw_text = self.get_data(
            file_path)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        raw_text = self.raw_text[idx]
        phones = np.array(text_to_sequence(self.text[idx]))
        return (basename, phones, raw_text)

    def get_data(self, file_name):
        return get_data(file_path=file_name)

    def collate_fn(self, data):
        ids = []
        speakers = []
        texts = []
        raw_texts = []
        text_lens = []
        for d in data:
            ids.append(d[0])
            speakers.append(d[1])
            texts.append(d[2])
            text_lens.append(d[2].shape[0])
            raw_texts.append(d[3])
        speakers = np.array(speakers)
        text_lens = np.array(text_lens)
        texts = pad_1D(texts)
        return ids, raw_texts, speakers, texts, text_lens, max(text_lens)
