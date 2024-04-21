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
    use_deepspeed=True, #def false
    speaker_file_path=SPEAKER_PATH
)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[SPEAKER_REFERENCE])

print("====================APPLICATION IS READY====================")


def text_to_speech(
        text,
        temperature=0.75,
        repetition_penalty=10.0,
        top_p=1.0,
        top_k=50
) -> bytes:
    print("Processing...")
    out = model.inference(
        text,
        language="en",
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        top_p=top_p,
        top_k=top_k
    )

    print("Extracting wav...")
    data = torch.tensor(out["wav"]).unsqueeze(0)

    print("Making wav...")
    return Audio._make_wav(data, 24000, False)


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

    def do_GET(self):
        self.send_response(HTTPStatus.OK)
        self.end_headers()


httpd = socketserver.TCPServer(('', 8002), Handler)
httpd.serve_forever()
