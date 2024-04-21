import http.server
import socketserver
import json
from http import HTTPStatus
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from IPython.display import Audio

# Add here the xtts_config path
CONFIG_PATH = "./resources/config.json"
# Add here the vocab file that you have used to train the model
TOKENIZER_PATH = "./resources/xtts/vocab.json"
# Add here the checkpoint that you want to do inference with
XTTS_CHECKPOINT = "./resources/xtts/model.pth"
SPEAKER_PATH = "./resources/xtts/speakers_xtts.pth"
# Add here the speaker reference
SPEAKER_REFERENCE = "./resources/samples/en_female_sample.wav"


print("Loading model...")
config = XttsConfig()
config.load_json(CONFIG_PATH)
model = Xtts.init_from_config(config)
model.load_checkpoint(
    config,
    checkpoint_path=XTTS_CHECKPOINT,
    vocab_path=TOKENIZER_PATH,
    use_deepspeed=False,
    speaker_file_path=SPEAKER_PATH
)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[SPEAKER_REFERENCE])

print("====================Application is ready====================")


def text_to_speech(
        text,
        temperature=0.7
) -> bytes:
    print("Inferencing...")
    out = model.inference(
        text,
        language="en",
        temperature=temperature,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding
    )
    print("Making wav...")
    return Audio._make_wav(torch.tensor(out["wav"]).unsqueeze(0), 24000, False)


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
        print('Получено сообщение: ', message)

        self._set_headers()
        try:
            wav = text_to_speech(
                message['text'],
                message.get("temperature", None)
            )
            self.wfile.write(wav)
        except KeyError as err:
            self.wfile.write(f"Ошибка, отсутствуют необходимые параметры в теле запроса: {err}".encode())

    def do_GET(self):
        self.send_response(HTTPStatus.OK)
        self.end_headers()


httpd = socketserver.TCPServer(('', 8002), Handler)
httpd.serve_forever()
