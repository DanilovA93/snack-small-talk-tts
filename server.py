import http.server
import socketserver
import json
from http import HTTPStatus
import TTSService


def text_to_speech(
        prompt,
        temperature=0.75,
        repetition_penalty=5.0
) -> bytes:
    return TTSService.process(
        prompt=prompt,
        temperature=temperature,
        repetition_penalty=repetition_penalty
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

        try:
            self._set_headers()
            wav = text_to_speech(
                message['prompt'],
                message.get("temperature", None),
                message.get("repetition_penalty", None)
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
