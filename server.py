import http.server
import socketserver
import json
from http import HTTPStatus
import coqui_tts as ctts
import speech_recognition as sr

tts = ctts.TextToSpeech()

def text_to_speech(speaker_id, text):
    return tts.convert(text)

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
