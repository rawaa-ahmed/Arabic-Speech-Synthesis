import mishkal.tashkeel
import numpy as np
import torch
from buckwalter import bw2ar
from phonetise.phonetise import phonetise
from text import text_to_sequence
from model_utils import get_model
from tools import synth_samples, to_device, get_vocoder
import constants

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_arabic_text(text, bw=False, ts=False):
    if bw:
        text = "".join([bw2ar[l] if l in bw2ar else l for l in text])
    if ts:
        vocalizer = mishkal.tashkeel.TashkeelClass()
        text = vocalizer.tashkeel(text).strip()
    phones = phonetise(text)[1]    
    print("Raw Text: {}".format(text))
    print("Phones: {}".format(phones))

    phones = "{" + ' '.join(phones) + "}"

    sequence = np.array(
        text_to_sequence(phones)
    )
    return np.array(sequence)
    

def prepare_tts_model(step=constants.RESTORE_STEP):
    """
    Prepares the TTS model at some step
    """
    # Getting model for inference
    model = get_model(step, DEVICE)

    # Loading vocoder
    vocoder = get_vocoder(DEVICE, constants.VOCODER_CONFIG_PATH, constants.VOCODER_PRETRAINED_MODEL_PATH)
    return model, vocoder

def infer_tts(
    text,
    model,
    vocoder,
    bw=True,
    apply_tshkeel=False,
    pitch_control=1.0,
    energy_control=1.0,
    duration_control=1.0,
    save_path='sample.wav'
):
    """
    Used to synthesize any input text 
    ----
    PARAMS
    text: text to convert to waveform
    pitch_control: to change the pitch of the audio as required
    save_path: the path where the waveform will be saved
    """
    
    ids = [text[:100]]
    raw_texts = [text[:100]]
    texts = np.array([preprocess_arabic_text(text, bw=bw, ts=apply_tshkeel)])
    text_lens = np.array([len(texts[0])])
    batchs = [(ids, raw_texts, np.array([0]), texts, text_lens, max(text_lens))]
    print('start synthesis...')

    for batch in batchs:
        batch = to_device(batch, DEVICE)
        with torch.no_grad():
            # Forward
            output = model(*(batch[2:]), p_control=pitch_control, e_control=energy_control, d_control=duration_control)
            synth_samples(
                batch,
                output,
                vocoder,
                save_path=save_path
            )
