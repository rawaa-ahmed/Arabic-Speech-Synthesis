from inference import infer_tts, prepare_tts_model
import constants
class TextToSpeech:
    def __init__(self,step=1045000):
        self.vocoder_config_path = constants.VOCODER_CONFIG_PATH
        self.speaker_pre_trained_path = constants.VOCODER_PRETRAINED_MODEL_PATH
        self.model, self.vocoder= prepare_tts_model(step)

    def synthesize( 
        self,
        text,
        bw=True,
        apply_tshkeel=False,
        pitch_control=1.0,
        energy_control=1.0,
        duration_control=1.0,
        save_path='sample.wav'
        ):
        if(text!=''):
            print("text:",text)
            print('bw:',bw)
            print("apply_tshkeel:",apply_tshkeel)
            print("pitch_control:",pitch_control)
            print("energy_control:",energy_control)
            print("duration_control:",duration_control)
            print("save_path:",save_path)
            model=self.model
            vocoder=self.vocoder
            infer_tts(text, model, vocoder, bw, apply_tshkeel, pitch_control, energy_control, duration_control, save_path)

