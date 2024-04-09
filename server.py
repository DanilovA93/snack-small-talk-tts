import http.server
import socketserver
import json
from http import HTTPStatus
import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Init TTS with the target model name
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def text_to_speech(speaker_id, text):
    wav = tts.tts(
        text=text,
        speaker_wav="./samples_en_sample.wav",
        language="en"
    )
    return wav

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
        wav = text_to_speech(message['speaker_id'], message['text'])
        self.wfile.write(wav)

    def do_GET(self):
        self.send_response(HTTPStatus.OK)
        self.end_headers()


httpd = socketserver.TCPServer(('', 8002), Handler)
httpd.serve_forever()
