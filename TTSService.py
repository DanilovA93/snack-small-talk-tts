import torch

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from IPython.display import Audio


CONFIG_PATH =               "./resources/config.json"
CHECKPOINT_PATH =           "./resources/xtts/model.pth"
TOKENIZER_PATH =            "./resources/xtts/vocab.json"
SPEAKER_PATH =              "./resources/xtts/speakers_xtts.pth"
SPEAKER_REFERENCE_PATH =    "./resources/samples/female_sample.wav"

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
    eval=True,
    # use_deepspeed=True,
)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
    audio_path=[SPEAKER_REFERENCE_PATH],
    # gpt_cond_len=30,
    # gpt_cond_chunk_len=4,
    # max_ref_length=60
)

print("GPT service is ready")


def process(
        prompt,
        temperature=0.75,
        repetition_penalty=5.0
) -> bytes:
    print("Processing...")
    out = model.inference(
        prompt,
        language=LANGUAGE,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=temperature,
        repetition_penalty=repetition_penalty
    )

    data = torch.tensor(out["wav"]).unsqueeze(0)

    return Audio._make_wav(data, 24000, False)
