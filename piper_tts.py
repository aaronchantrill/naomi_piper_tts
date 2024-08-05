import json
import logging
import os
import struct
from collections import OrderedDict
from naomi import app_utils
from naomi import paths
from naomi import plugin
from naomi import profile
from piper.voice import PiperVoice

class PiperTTSPlugin(plugin.TTSPlugin):
    """
    https://github.com/rhasspy/piper
    https://github.com/rhasspy/piper/blob/master/VOICES.md
    """
    voices = {
        "de-DE": {
            "eva_k": {
                "model_url": 'https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/eva_k/x_low/de_DE-eva_k-x_low.onnx?download=true',
                "config_url": 'https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/eva_k/x_low/de_DE-eva_k-x_low.onnx.json?download=true.json',
                "model_file": "de_DE-eva_k-x_low.onnx",
            },
            "karlsson": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/karlsson/low/de_DE-karlsson-low.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/karlsson/low/de_DE-karlsson-low.onnx.json?download=true.json",
                "model_file": "de_DE-karlsson-low.onnx"
            }
        },
        "en-US": {
            "amy_low": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/low/en_US-amy-low.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/low/en_US-amy-low.onnx.json?download=true.json",
                "model_file": "en_US-amy-low.onnx"
            },
            "amy_medium": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx.json?download=true.json",
                "model_file": "en_US-amy-medium.onnx"
            },
            "arctic": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/arctic/medium/en_US-arctic-medium.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/arctic/medium/en_US-arctic-medium.onnx.json?download=true.json",
                "model_file": "en_US-arctic-medium.onnx"
            },
            "bryce": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/bryce/medium/en_US-bryce-medium.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/bryce/medium/en_US-bryce-medium.onnx.json?download=true.json",
                "model_file": "en_US-bryce-medium.onnx"
            },
            "danny": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/danny/low/en_US-danny-low.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/danny/low/en_US-danny-low.onnx.json?download=true.json",
                "model_file": "en_US-danny-low.onnx"
            },
            "hfc_female": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx.json?download=true.json",
                "model_file": "en_US-hfc_female-medium.onnx"
            },
            # https://github.com/dnhkng/GlaDOS
            "glados":{
                "model_url": "https://raw.githubusercontent.com/dnhkng/GlaDOS/main/models/glados.onnx",
                "config_url": "https://raw.githubusercontent.com/dnhkng/GlaDOS/main/models/glados.onnx.json",
                "model_file": "glados.onnx"
            }
        }
    }

    def __init__(self, *args, **kwargs):
        plugin.TTSPlugin.__init__(self, *args, **kwargs)
        self.voice = profile.get(['piper-tts', 'voice'])
        self.speaker = profile.get(['piper-tts', 'speaker'])
        self.speaker_id = self.get_speaker_id(
            self.voice,
            self.speaker
        )
        self.load_model()

    def load_model(self):
        locale = profile.get(['language'])
        model_dir = os.path.join(paths.sub('piper'), locale, self.voice)
        model_file = os.path.join(model_dir, self.voices[profile.get("language")][self.voice]['model_file'])
        config_file = f"{model_file}.json"
        self.pipervoice = PiperVoice.load(model_file)
        # Get the sample rate from the config file
        try:
            with open(config_file) as f:
                config = json.load(f)
            self.sample_rate = config['audio']['sample_rate']
        except Exception as e:
            print(e)
            self.sample_rate = 22050

    def settings(self):
        return OrderedDict(
            [
                (
                    ('piper-tts', 'voice'), {
                        'type': 'listbox',
                        'title': self.gettext('Voice for Piper Text to Speech'),
                        'description': " ".join([
                            self.gettext("This is the voice file I will use to speak to you."),
                            self.gettext("A voice file may contain multiple speakers."),
                            self.gettext("The speakers within a voice may be quite distinct."),
                            self.gettext("A single voice can include both male and female speakers.")
                        ]),
                        'options': self.get_voices,
                        'default': 'arctic'
                    }
                ),
                (
                    ('piper-tts', 'speaker'), {
                        'type': 'listbox',
                        'title': self.gettext('Speaker for Piper Text to Speech'),
                        'description': " ".join([
                            self.gettext('This is the speaker voice I will use to speak to you')
                        ]),
                        'options': self.get_speakers
                    }
                )
            ]
        )

    def get_voices(self):
        """
        List of voices
        """
        # In the language variable, the language and region are separated by
        # a dash. Mimic3 uses an underscore. So convert any dashes to
        # underscores
        locale = profile.get(['language'])
        return [voice for voice in self.voices[locale]]

    def install_voice(self, locale, voice):
        model_dir = os.path.join(paths.sub('piper'), locale, voice)
        os.makedirs(model_dir, exist_ok=True)
        model_url = self.voices[locale][voice]['model_url']
        model_filename = os.path.join(model_dir, self.voices[locale][voice]['model_file'])
        print(f"Downloading from {model_url} to {model_filename}")
        app_utils.download_file(model_url, model_filename)
        config_url = self.voices[locale][voice]['config_url']
        config_filename = os.path.join(model_dir, f"{self.voices[locale][voice]['model_file']}.json")
        print(f"Downloading from {config_url} to {config_filename}")
        app_utils.download_file(config_url, config_filename)

    def get_speakers(self):
        """
        List of speakers is kept in the yaml profile
        """
        locale = profile.get(['language'])
        voice = profile.get(['piper-tts', 'voice'])
        if voice is None:
            return ['Default']
        model_dir = os.path.join(paths.sub('piper'), locale, voice)
        model_filename = os.path.join(model_dir, self.voices[locale][voice]['model_file'])
        config_filename = os.path.join(model_dir, f"{model_filename}.json")
        if not os.path.isfile(config_filename):
            # We have to download at least the config file to get a list of voices
            # As this list is not very descriptive, we might as well go ahead
            # and install the voice, since the next step will be to at least
            # test the voice
            self.install_voice(
                locale,
                voice
            )
        with open(config_filename) as f:
            config = json.load(f)
        speaker_id_map = config['speaker_id_map']
        if len(speaker_id_map):
            speakers = [speaker for speaker in speaker_id_map]
        else:
            speakers = ['Default']
        return speakers

    def get_speaker_id(self, voice, speaker):
        """
        Convert a speaker string to an id
        If the
        """
        locale = profile.get(['language'])
        model_dir = os.path.join(paths.sub('piper'), locale, voice)
        model_filename = os.path.join(model_dir, self.voices[locale][voice]['model_file'])
        config_filename = os.path.join(model_dir, f"{model_filename}.json")
        speaker_id = None
        if os.path.isfile(config_filename):
            with open(config_filename) as f:
                config = json.load(f)
            speaker_id_map = config['speaker_id_map']
            if speaker in speaker_id_map:
                speaker_id = speaker_id_map[speaker]
        return speaker_id

    # This plugin can receive a voice as a third parameter. This allows easier
    # testing of different voices.
    def say(self, phrase, voice=None):
        # Any upper-case words will be spelled rather than read. For instance:
        # "nasa" will be read "na-saw" and "NASA" will be read "EN AY ES AY"
        # Also, a word with numbers in it will only be read up to the first
        # number, so for words like "MyWifi5248Network" we need to split the
        # string into letters and numbers
        reconfigure = False
        if voice:
            if "#" in voice:
                # split voice and speaker at '#'
                voice, speaker = voice.split('#')
                if voice != self.voice:
                    self.voice = voice
                    self.load_model()
                speaker_id = get_speaker_id(voice, speaker)
                if speaker_id != self.speaker_id:
                    self.speaker_id = speaker_id
        output = self.pipervoice.synthesize_stream_raw(phrase, self.speaker_id)
        byte_output = b''
        for block in output:
            byte_output += block
        # Get the sample rate from the config file
        return self.pcm2wav(byte_output, sample_rate=self.sample_rate, bits_per_sample=16, channels=1)

    # https://stackoverflow.com/questions/67317366/how-to-add-header-info-to-a-wav-file-to-get-a-same-result-as-ffmpeg
    # https://stackoverflow.com/questions/28137559/can-someone-explain-wavwave-file-headers
    @staticmethod
    def pcm2wav(audio, sample_rate=22050, bits_per_sample=16, channels=1):
        channels = 1
        bits_per_sample = 16
        if audio.startswith("RIFF".encode()):
            return audio
        else:
            sampleNum = len(audio)
            rHeaderInfo = "RIFF".encode()                    #  1- 4 - "RIFF"
            rHeaderInfo += struct.pack('i', sampleNum + 44)  #  5- 8 - File size
            rHeaderInfo += 'WAVEfmt '.encode()               #  9-16 - "WAVEfmt "
            rHeaderInfo += struct.pack('i', bits_per_sample) # 17-20 - bits per sample
            rHeaderInfo += struct.pack('h', 1)               # 21-22 - WAVE_FORMAT_PCM
            rHeaderInfo += struct.pack('h', channels)        # 23-24 - Channels
            rHeaderInfo += struct.pack('i', sample_rate)     # 25-26 - Sample Rate
            rHeaderInfo += struct.pack('i', sample_rate * int(bits_per_sample * channels / 8))
            rHeaderInfo += struct.pack("h", int(bits_per_sample * channels / 8))
            rHeaderInfo += struct.pack("h", bits_per_sample) # 35-36 - Bits per sample
            rHeaderInfo += "data".encode()                   # 37-40 - 'data'
            rHeaderInfo += struct.pack('i', sampleNum)       # 41-44 - file size
            rHeaderInfo += audio
            return rHeaderInfo

