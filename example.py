from speech_synthesis import TextToSpeech
from IPython.display import Audio

model = TextToSpeech()

text="وَسادْيُو مَانِي فِي الدَّقِيقَتَيْنِ السّابِعَةَ عَشْرَةَ - وَ الْخَامِسَةَ وَالْأَرْبعينَ مِنَ الشَّوْطِ الْأَوَّلِ لِلْمُبَارَاةِ"
model.synthesis(text,save_path="sample.wav")

# Audio("sample.wav")