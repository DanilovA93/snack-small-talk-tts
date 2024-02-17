import http.server
import socketserver
import json
from http import HTTPStatus
import torch
from IPython.display import Audio
from nemo.collections.tts.models.base import SpectrogramGenerator, Vocoder

# Select Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load spectrogram generator
from nemo.collections.tts.models import FastPitchModel
spec_generator = FastPitchModel.from_pretrained("tts_en_fastpitch_multispeaker").eval().to(device)

# Load Vocoder
from nemo.collections.tts.models import HifiGanModel
model = HifiGanModel.from_pretrained(model_name="tts_en_hifitts_hifigan_ft_fastpitch")

def text_to_speech(speaker_id, text):
    tokens = spec_generator.parse(text, normalize=False)
    spectrogram = spec_generator.generate_spectrogram(tokens=tokens, speaker=speaker_id)
    audio = model.convert_spectrogram_to_audio(spec=spectrogram)
    audio = audio.cpu().detach().numpy()[0]
    return Audio._make_wav(audio, 44100, False)


class Handler(http.server.SimpleHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-type', 'audio/x-wav')
        # Allow requests from any origin, so CORS policies don't
        # prevent local development.
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

    def do_POST(self):
        content_len = int(self.headers.get('Content-Length'))
        message = json.loads(self.rfile.read(content_len))
        self._set_headers()
        print('Rq message: ', message)
        self.wfile.write(text_to_speech(message['speaker_id'], message['text']))

    def do_GET(self):
        self.send_response(HTTPStatus.OK)
        self.end_headers()


httpd = socketserver.TCPServer(('', 8000), Handler)
httpd.serve_forever()
