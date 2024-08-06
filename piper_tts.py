import json
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
            },
            "kerstin": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/kerstin/low/de_DE-kerstin-low.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/kerstin/low/de_DE-kerstin-low.onnx.json?download=true.json",
                "model_file": "de_DE-kerstin-low.onnx"
            },
            "mls": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/mls/medium/de_DE-mls-medium.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/mls/medium/de_DE-mls-medium.onnx.json?download=true.json",
                "model_file": "de_DE-mls-medium.onnx"
            },
            "pavoque": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/pavoque/low/de_DE-pavoque-low.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/pavoque/low/de_DE-pavoque-low.onnx.json?download=true.json",
                "model_file": "de_DE-pavoque-low.onnx"
            },
            "ramona": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/ramona/low/de_DE-ramona-low.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/ramona/low/de_DE-ramona-low.onnx.json?download=true.json",
                "model_file": "de_DE-ramona-low.onnx"
            },
            "thorsten_low": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/thorsten/low/de_DE-thorsten-low.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/thorsten/low/de_DE-thorsten-low.onnx.json?download=true.json",
                "model_file": "de_DE-thorsten-low.onnx"
            },
            "thorsten_medium": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx.json?download=true.json",
                "model_file": "de_DE-thorsten-medium.onnx"
            },
            "thorsten_high": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx.json?download=true.json",
                "model_file": "de_DE-thorsten-high.onnx"
            },
            "thorsten_emotional": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/thorsten_emotional/medium/de_DE-thorsten_emotional-medium.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/thorsten_emotional/medium/de_DE-thorsten_emotional-medium.onnx.json?download=true.json",
                "model_file": "de_DE-thorsten_emotional-medium.onnx"
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
            "hfc_male": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/hfc_male/medium/en_US-hfc_male-medium.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/hfc_male/medium/en_US-hfc_male-medium.onnx.json?download=true.json",
                "model_file": "en_US-hfc_male-medium.onnx"
            },
            "joe": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/joe/medium/en_US-joe-medium.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/joe/medium/en_US-joe-medium.onnx.json?download=true.json",
                "model_file": "en_US-joe-medium.onnx"
            },
            "john": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/john/medium/en_US-john-medium.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/john/medium/en_US-john-medium.onnx.json?download=true.json",
                "model_file": "en_US-john-medium.onnx"
            },
            "kathleen": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/kathleen/low/en_US-kathleen-low.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/kathleen/low/en_US-kathleen-low.onnx.json?download=true.json",
                "model_file": "en_US-kathleen-low.onnx"
            },
            "kristin": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/kristin/medium/en_US-kristin-medium.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/kristin/medium/en_US-kristin-medium.onnx.json?download=true.json",
                "model_file": "en_US-kristin-medium.onnx"
            },
            "kusal": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/kusal/medium/en_US-kusal-medium.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/kusal/medium/en_US-kusal-medium.onnx.json?download=true.json",
                "model_file": "en_US-kusal-medium.onnx"
            },
            "l2arctic": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/l2arctic/medium/en_US-l2arctic-medium.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/l2arctic/medium/en_US-l2arctic-medium.onnx.json?download=true.json",
                "model_file": "en_US-l2arctic-medium.onnx"
            },
            "lessac_low": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/low/en_US-lessac-low.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/low/en_US-lessac-low.onnx.json?download=true.json",
                "model_file": "en_US-lessac-low.onnx"
            },
            "lessac_medium": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json?download=true.json",
                "model_file": "en_US-lessac-medium.onnx"
            },
            "lessac_high": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/high/en_US-lessac-high.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/high/en_US-lessac-high.onnx.json?download=true.json",
                "model_file": "en_US-lessac-high.onnx"
            },
            "libritts": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/libritts/high/en_US-libritts-high.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/libritts/high/en_US-libritts-high.onnx.json?download=true.json",
                "model_file": "en_US-libritts-high.onnx"
            },
            "libritts_r": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx.json?download=true.json",
                "model_file": "en_US-libritts_r-medium.onnx"
            },
            "ljspeech_medium": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx.json?download=true.json",
                "model_file": "en_US-ljspeech-medium.onnx"
            },
            "ljspeech_high": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ljspeech/high/en_US-ljspeech-high.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ljspeech/high/en_US-ljspeech-high.onnx.json?download=true.json",
                "model_file": "en_US-ljspeech-high.onnx"
            },
            "norman": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/norman/medium/en_US-norman-medium.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/norman/medium/en_US-norman-medium.onnx.json?download=true.json",
                "model_file": "en_US-norman-medium.onnx"
            },
            "ryan_low": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/low/en_US-ryan-low.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/low/en_US-ryan-low.onnx.json?download=true.json",
                "model_file": "en_US-ryan-low.onnx"
            },
            "ryan_medium": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/medium/en_US-ryan-medium.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/low/en_US-ryan-low.onnx.json?download=true.json",
                "model_file": "en_US-ryan-medium.onnx"
            },
            "ryan_high": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high/en_US-ryan-high.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high/en_US-ryan-high.onnx.json?download=true.json",
                "model_file": "en_US-ryan-high.onnx"
            },
            # https://github.com/dnhkng/GlaDOS
            "glados": {
                "model_url": "https://raw.githubusercontent.com/dnhkng/GlaDOS/main/models/glados.onnx",
                "config_url": "https://raw.githubusercontent.com/dnhkng/GlaDOS/main/models/glados.onnx.json",
                "model_file": "glados.onnx"
            }
        },
        "fr-FR": {
            "gilles": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/fr/fr_FR/gilles/low/fr_FR-gilles-low.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/fr/fr_FR/gilles/low/fr_FR-gilles-low.onnx.json?download=true.json",
                "model_file": "fr_FR-gilles-low.onnx"
            },
            "mls": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/fr/fr_FR/mls/medium/fr_FR-mls-medium.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/fr/fr_FR/mls/medium/fr_FR-mls-medium.onnx.json?download=true.json",
                "model_file": "fr_FR-mls-medium.onnx"
            },
            "mls_1840": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/fr/fr_FR/mls_1840/low/fr_FR-mls_1840-low.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/fr/fr_FR/mls_1840/low/fr_FR-mls_1840-low.onnx.json?download=true.json",
                "model_file": "fr_FR-mls_1840-low.onnx"
            },
            "siwis_low": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/fr/fr_FR/siwis/low/fr_FR-siwis-low.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/fr/fr_FR/siwis/low/fr_FR-siwis-low.onnx.json?download=true.json",
                "model_file": "fr_FR-siwis-low.onnx"
            },
            "siwis_medium": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx.json?download=true.json",
                "model_file": "fr_FR-siwis-medium.onnx"
            },
            "tom": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/fr/fr_FR/tom/medium/fr_FR-tom-medium.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/fr/fr_FR/tom/medium/fr_FR-tom-medium.onnx.json?download=true.json",
                "model_file": "fr_FR-tom-medium.onnx"
            },
            "upmc": {
                "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx?download=true",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx.json?download=true.json",
                "model_file": "fr_FR-upmc-medium.onnx"
            }
        }
    }

    def __init__(self, *args, **kwargs):
        plugin.TTSPlugin.__init__(self, *args, **kwargs)
        self.voice = profile.get(['piper-tts', 'voice'])
        self.current_voice = self.voice
        self.speaker = profile.get(['piper-tts', 'speaker'])
        self.current_speaker = self.speaker
        self.speaker_id = self.get_speaker_id(
            self.voice,
            self.speaker
        )
        self.current_speaker_id = self.speaker_id
        self.sample_rate = {}
        self.load_model(self.voice)

    def load_model(self, voice):
        locale = profile.get(['language'])
        self.install_voice(locale, voice)
        model_dir = os.path.join(paths.sub('piper'), locale, voice)
        model_file = os.path.join(model_dir, self.voices[profile.get("language")][voice]['model_file'])
        config_file = f"{model_file}.json"
        self.pipervoice = PiperVoice.load(model_file)
        # Get the sample rate from the config file
        try:
            with open(config_file) as f:
                config = json.load(f)
            self.sample_rate[voice] = config['audio']['sample_rate']
        except Exception as e:
            print(e)
            self.sample_rate[voice] = 22050

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
        if not os.path.isfile(model_filename):
            app_utils.download_file(model_url, model_filename)
        config_url = self.voices[locale][voice]['config_url']
        config_filename = os.path.join(model_dir, f"{self.voices[locale][voice]['model_file']}.json")
        if not os.path.isfile(config_filename):
            app_utils.download_file(config_url, config_filename)

    def get_speakers(self, voice=None):
        """
        List of speakers is kept in the yaml profile
        """
        locale = profile.get(['language'])
        if voice is None:
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
        if voice:
            if "#" in voice:
                # split voice and speaker at '#'
                voice, speaker = voice.split('#')
                if voice != self.voice:
                    self.load_model(voice)
                    self.current_voice = voice
                speaker_id = self.get_speaker_id(voice, speaker)
                if speaker_id != self.speaker_id:
                    self.current_speaker_id = speaker_id
            else:
                if voice != self.current_voice:
                    self.load_model(voice)
                    self.current_voice = voice
                speaker_id = None
        else:
            # If a voice is not passed in, use the default voice
            voice = self.voice
            if voice != self.current_voice:
                self.load_model(voice)
                self.current_voice = voice
            speaker_id = self.speaker_id
        output = self.pipervoice.synthesize_stream_raw(phrase, speaker_id)
        byte_output = b''
        for block in output:
            byte_output += block
        # Get the sample rate from the config file
        return self.pcm2wav(byte_output, sample_rate=self.sample_rate[voice], bits_per_sample=16, channels=1)

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

