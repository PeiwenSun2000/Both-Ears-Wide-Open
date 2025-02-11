import argparse
import gc
import torch
import torchaudio
from einops import rearrange
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.models.pretrained import get_pretrained_model
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.training.utils import copy_state_dict
import json

model = None

def load_model(model_config=None, model_ckpt_path=None, pretrained_name=None, device="cuda", model_half=False):
    global model, sample_rate, sample_size
    
    if pretrained_name is not None:
        print(f"Loading pretrained model {pretrained_name}")
        model, model_config = get_pretrained_model(pretrained_name)
    elif model_config is not None and model_ckpt_path is not None:
        print(f"Creating model from config")
        model = create_model_from_config(model_config)
        print(f"Loading model checkpoint from {model_ckpt_path}")
        load_state = torch.load(model_ckpt_path)
        load_state = {key.replace("diffusion.", ""): value for key, value in load_state.items()}
        copy_state_dict(model, load_state)

    model.to(device).eval().requires_grad_(False)

    if model_half:
        model.to(torch.float16)
        
    print(f"Done loading model")
    return model, model_config

def generate_audio(prompt, model, device):
    resampler = torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000) 
    conditioning = [{"prompt": prompt, "seconds_start": 0, "seconds_total": 10}]

    audio = generate_diffusion_cond(
        model, 
        conditioning=conditioning,
        negative_conditioning=None,
        steps=250,
        cfg_scale=7.5,
        batch_size=1,
        sample_size=int(44100*10),
        sample_rate=44100,
        seed=-1,
        device=device,
        sampler_type="dpmpp-3m-sde",
        sigma_min=0.03,
        sigma_max=1000,
        init_audio=None,
        init_noise_level=1.0,
        mask_args=None,
        callback=None,
        scale_phi=0
    )

    # For fair comparison, downsample to 16k
    # print(audio.shape)
    torchaudio.save("output.wav",  resampler(audio.squeeze(0).cpu()).squeeze(0), sample_rate=16000)
    print("Generated audio saved to output.wav")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate spatial audio from a text prompt.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for audio generation.")
    parser.add_argument("--bewo", type=bool, default=True, help="Use a bewo model.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on.")
    args = parser.parse_args()


    if not args.bewo:
        model, model_config = load_model(pretrained_name="stabilityai/stable-audio-open-1.0", device=args.device)
    else:
        # Example paths, replace with actual paths if needed
        model_config_path = "bewo_config/model_config_sim.json"
        model_ckpt_path = "bewo_config/BEWO_nl.ckpt"

        with open(model_config_path) as f:
            model_config = json.load(f)
        model, model_config = load_model(model_config=model_config, model_ckpt_path=model_ckpt_path, device=args.device)
    generate_audio(args.prompt, model, args.device)
