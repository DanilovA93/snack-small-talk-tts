import http.server
import socketserver
import json
from http import HTTPStatus
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio
from transformers import AutoProcessor, AutoModel

# download and load all models
processor = AutoProcessor.from_pretrained("suno/bark-small")
model = AutoModel.from_pretrained("suno/bark-small").to("cuda")

def text_to_speech(speaker_id, text):
    inputs = processor(
        text=[text],
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    speech_values = model.generate(**inputs, do_sample=True)
    sampling_rate = model.generation_config.sample_rate
    return Audio._make_wav(speech_values.cpu().numpy().squeeze(), rate=sampling_rate, False)


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


httpd = socketserver.TCPServer(('', 8002), Handler)
httpd.serve_forever()
