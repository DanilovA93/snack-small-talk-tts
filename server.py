import http.server
import socketserver
import json
from http import HTTPStatus
import TTSService


def text_to_speech(
        text,
        temperature=0.75,
        repetition_penalty=10.0,
        top_p=1.0,
        top_k=50
) -> bytes:
    return TTSService.process(
        text=text,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        top_p=top_p,
        top_k=top_k
    )


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
        print('Rq body: ', message)

        self._set_headers()
        try:
            wav = text_to_speech(
                message['text'],
                message.get("temperature", None),
                message.get("repetition_penalty", None),
                message.get("top_p", None),
                message.get("top_k", None)
            )
            self.wfile.write(wav)
        except KeyError as err:
            self.wfile.write(f"Error, required parameters are missing in the request body: {err}".encode())
        except Exception as err:
            message = f"Error: {err}"
            print(message)
            self.wfile.write(message.encode())

    def do_GET(self):
        self.send_response(HTTPStatus.OK)
        self.end_headers()


httpd = socketserver.TCPServer(('', 8002), Handler)
httpd.serve_forever()
