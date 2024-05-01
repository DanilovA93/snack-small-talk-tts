import os
import torch

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from IPython.display import Audio


CONFIG_PATH =               "./resources/config.json"
CHECKPOINT_PATH =           "./resources/xtts/model.pth"
TOKENIZER_PATH =            "./resources/xtts/vocab.json"
SPEAKER_PATH =              "./resources/xtts/speakers_xtts.pth"
SPEAKER_REFERENCE_PATH =    "./resources/samples/female_sample.wav"

CACHE_DIR = "./cache/"
LANGUAGE = "en"

print("Reading config...")
config = XttsConfig()
config.load_json(CONFIG_PATH)

print("Loading model...")
model = Xtts.init_from_config(config)
model.load_checkpoint(
    config,
    checkpoint_path=CHECKPOINT_PATH,
    vocab_path=TOKENIZER_PATH,
    speaker_file_path=SPEAKER_PATH,
    eval=True
)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[SPEAKER_REFERENCE_PATH])

print("TTS service is ready")


def process(
        prompt,
        temperature=0.75,
        repetition_penalty=5.0
) -> bytes:
    print("Processing...")

    cached_response = get_from_cache(prompt)
    if cached_response is not None:
        return cached_response
    else:
        output = model.inference(
            prompt,
            language=LANGUAGE,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            do_sample=True
        )
        data = torch.tensor(output["wav"]).unsqueeze(0)
        audio = Audio._make_wav(data, 24000, False)
        cache(prompt, audio)
        return audio


def get_from_cache(request):
    filename = str(hash(request))
    path_to_file = CACHE_DIR + filename

    if os.path.isfile(path_to_file):
        with open(path_to_file, "rb") as f:
            output = f.read()
            f.close()
            return output
    else:
        return None


def cache(request, response):
    filename = str(hash(request))
    path_to_file = CACHE_DIR + filename

    file = open(path_to_file, "wb")
    file.write(response)
    file.close()
