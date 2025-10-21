import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


CONFIG_PATH = "./model/config.json"
XTTS_PATH = "./model/model.pth"

TOKENIZER_PATH = "./model/vocab.json"

SPEAKER_REFERENCE_WAV_PATH = "./speaker_reference/reference.wav"
OUTPUT_WAV_PATH = "./output.wav"


def main(text:str, language: str):

    
    print("Initializing the XTTS model from configuration.")
    config = XttsConfig()
    config.load_json(CONFIG_PATH)
    model = Xtts.init_from_config(config)
    
    print("Model loading from checkpoint...")
    model.load_checkpoint(
        config, 
        checkpoint_path=str(XTTS_PATH), 
        vocab_path=str(TOKENIZER_PATH), 
        speaker_file_path=" ", 
        use_deepspeed=False,
        eval=True)
    
    print("Model loaded successfully.")

    print("Computing speaker latents...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[SPEAKER_REFERENCE_WAV_PATH],
        gpt_cond_len=30,
        gpt_cond_chunk_len=4,
        max_ref_length=60
    )


    print("Inference...") 
    out = model.inference(
        text= text,
        language= language,
        gpt_cond_latent= gpt_cond_latent,
        speaker_embedding= speaker_embedding,
        repetition_penalty= 5.0,
        temperature= 0.75,
    )


    torchaudio.save(str(OUTPUT_WAV_PATH), torch.tensor(out["wav"]).unsqueeze(0), 24000)
    print(f"Output audio file '{OUTPUT_WAV_PATH}' saved")


if __name__ == "__main__":
    text = "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent."
    language = "en"
    main(text, language)
