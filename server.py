import http.server
import socketserver
import json
from http import HTTPStatus
import torch
from TTS.api import TTS
from IPython.display import Audio

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Init TTS with the target model name
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


def text_to_speech(text, emotion, rate = 24000) -> bytes:
    wav = tts.tts(
        text=text,
        speaker="Ana Florence",
        language="en",
        speaker_wav="./samples_en_sample.wav",
        emotion=emotion
    )
    print(1)
    audio = Audio._make_wav(wav, rate, False)
    print(2)
    return audio


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
        print('Тело запроса: ', message)

        self._set_headers()
        try:
            wav = text_to_speech(
                message['text'],
                message.get("emotion", 'Neutral'),
                message.get("rate", None)
            )
            self.wfile.write(wav)
        except KeyError as err:
            self.wfile.write(f"Ошибка, отсутствуют необходимые параметры в теле запроса: {err}".encode())

    def do_GET(self):
        self.send_response(HTTPStatus.OK)
        self.end_headers()


httpd = socketserver.TCPServer(('', 8002), Handler)
httpd.serve_forever()
