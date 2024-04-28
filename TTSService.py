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
    speaker_file_path=SPEAKER_PATH,
    eval=True,
    use_deepspeed=True,
).half()
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[SPEAKER_REFERENCE])


def process(
        prompt,
        temperature=0.75,
        repetition_penalty=5.0
) -> bytes:
    print("Processing...")
    out = model.inference(
        prompt,
        language="en",
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        repetition_penalty=repetition_penalty
    )

    print("Extracting wav...")
    data = torch.tensor(out["wav"]).unsqueeze(0)

    print("Making wav...")
    return Audio._make_wav(data, 24000, False)
