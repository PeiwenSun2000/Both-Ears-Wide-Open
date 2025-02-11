from openai import AzureOpenAI,OpenAI
import time
import json
import argparse
import gc
import torch
import torchaudio
from einops import rearrange
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.models.pretrained import get_pretrained_model
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.training.utils import copy_state_dict, copy_state_dict_naive
import json
# Set the API key and model name
def get_gpt_induction(input_text):
    model_select = "deepseek"
    if model_select == "openai":
        MODEL = "gpt-4o"
        client = AzureOpenAI(
            azure_endpoint="Your_end_point",
            api_version="Your_api_version",
            api_key="Your_api_key"
        )
    elif model_select == "deepseek":
        MODEL = "deepseek-chat"
        client = OpenAI(api_key="Your_api_key", base_url="https://api.deepseek.com"
        )

    def chat_completion_with_retry(client, model, messages, temperature, max_retries=3, retry_delay=0.5):
        retries = 0
        while retries < max_retries:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
                if response.choices[0].message.content[:7] == "```json":
                    # print(response.choices[0].message.content[7:-4])
                    json_data=json.loads(response.choices[0].message.content[8:-4])
                return json_data
            except Exception as e:
                retries += 1
                print(f"Error: {e}")
                print(f"Request failed. Retrying ({retries}/{max_retries})...")
                time.sleep(retry_delay)
        raise Exception("Maximum number of retries reached. Unable to complete the request.")

    template = """
    I will provide you with a caption. You have to return attributes based on my input. Please follow the procedures step-by-step strictly!

    1) Determine if the entire scene is likely to produce sound. Based on the input, identify the objects that may produce sound. If one object can make a sound, you can think it can sound. If impossible, skip other steps. - Sounding choices: `0: impossible`, `1: possible`
    2) Assess the size of the scene in which the audio occurs from: - Scene size choices: `1:outdoors`, `2:large`, `3:moderate`, `4:small`.
    3) Identify objects' sound descriptions. Then determine the position and basic descriptions for each object based on the input. You should consider both direction and distance: - direction choices: `1:left`, `2:front left`, `3:directly front`, `4:front right`, `5:right`. - Distance choices: `1:far`, `2:moderate`, `3:near`.
    4) Identify if the object is moving or not. If not, skip step 5. - Moving choices: `0: No moving`, `1: Moving`
    5) If some objects move, choose an end position that is different from the initial direction, and also choose a speed for the object from: - Speed choices: `1:slow`, `2:moderate`, `3:fast`.
    6) For both initial direction and end direction, note that you can return a decimal number when input involves a precise description of the angle. e.g., if the input is "a dog barks at front left 70 degree”, you can return `init_direction": 2.7, or `init_direction": 2.8, and you can infer what makes sense.
    7) Ensure that the positions, movements, and scene size you choose correspond realistically with objects in the real world and match the input.
    8) Provide your response in JSON format beginning with `{"` like the examples below. Example: with input as: "The bus engine idles on the left and a woman walks from right to front slowly." You should respond:
    The only return should like this:
    {
        "sound": 1,
        "size": 1, 
        "objects": {
        "Bus": {
            "init_direction": 1,
            "init_dis": 1,
            "moving": 0
        },
        "Woman": {
            "init_direction": 5,
            "init_dis": 1,
            "moving": 1,
            "end_direction": 3,
            "end_dis": 2,
            "speed": 1
        }
        }
    }
    """

    try:
        ret = chat_completion_with_retry(
            client,
            model=MODEL,
            messages=[
                {"role": "system", "content": "As an experienced text analyst. Given the text of a scenery, you need to evaluate spatial audio, including positions, movements, and the scene size. Your response should follow the specific guidelines below for clarity and accuracy:"},
                {"role": "user", "content": f"{template}\n{input_text}"}
            ],
            temperature=0.0,
            max_retries=3,
            retry_delay=0.5
        )
        print(ret)
        return ret
    except Exception as e:
        print(f"Error: {e}")


def load_model(model_config=None, model_ckpt_path=None, pretrained_name=None, device="cuda", model_half=False):
    
    if pretrained_name is not None:
        print(f"Loading pretrained model {pretrained_name}")
        model, model_config = get_pretrained_model(pretrained_name)
    elif model_config is not None and model_ckpt_path is not None:
        print(f"Creating model from config")
        model = create_model_from_config(model_config)
        load_state = torch.load(model_ckpt_path)
        # print(load_state.keys())
        load_state = {key.replace("diffusion.", ""): value for key, value in load_state.items()}
        copy_state_dict_naive(model, load_state)
    model.to(device).eval().requires_grad_(False)

    if model_half:
        model.to(torch.float16)
    # print(model.state_dict()["conditioner.conditioners.position.cross_a2t.embedding.weight"].cpu() == load_state["conditioner.conditioners.position.cross_a2t.embedding.weight"].cpu())
    print(f"Done loading model")
    return model, model_config

def generate_audio(prompt, model, device,):
    induction = get_gpt_induction(prompt)
    print(induction)
    position = []
    FINE_or_COARSE = 1
    for key,value in induction["objects"].items():
        if value["moving"] == 0:
            position.append((value["init_direction"],value["init_direction"],value["moving"],FINE_or_COARSE))
        else:
            moving_dict = {0:0,1:0.3,2:0.6,3:0.9}
            position.append((value["init_direction"],value["end_direction"],moving_dict[value["moving"]],FINE_or_COARSE))
    print(position)
    resampler = torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000) 
    conditioning = [{"prompt": prompt, "seconds_start": 0, "seconds_total": 10, "position":position}]

    audio = generate_diffusion_cond(
        model, 
        conditioning=conditioning,
        negative_conditioning=None,
        steps=250,
        cfg_scale=7.5,
        batch_size=1,
        # sample_size=441000,
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

def generate_audio_manual(prompt, model, device, init_direction,final_direction,moving):
    position = []
    FINE_or_COARSE = 1
    position.append((init_direction,final_direction,moving,FINE_or_COARSE))
    if position[0][2] == 0:
        pass
    else:
        moving_dict = {0:0,1:0.9,2:0.6,3:0.3}
        position = [(position[0][0],position[0][1],moving_dict[position[0][2]],FINE_or_COARSE)]
    # print(position)
    resampler = torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000) 
    conditioning = [{"prompt": prompt, "seconds_start": 0, "seconds_total": 10, "position":position}]
    print("----Generating Audio----")
    audio = generate_diffusion_cond(
        model,
        conditioning=conditioning,
        negative_conditioning=None,
        steps=250,
        cfg_scale=7.5,
        batch_size=1,
        # sample_size=441000,
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
    parser.add_argument("--manual", type=bool, default=False, help="Use manual position.")
    parser.add_argument("--init_direction", type=float, default=1, help="Initial direction from 1 (left) to 5 (right).")
    parser.add_argument("--final_direction", type=float, default=1, help="Final distance from 1 (left) to 5 (right).")
    parser.add_argument("--moving", type=float, default=0, help="Moving state. 0: No moving, 1: Moving slowly, 2: Moving moderately, 3: Moving fast")
    args = parser.parse_args()

    model = None

    if not args.bewo:
        model, model_config = load_model(pretrained_name="stabilityai/stable-audio-open-1.0", device=args.device)
    else:
        # Example paths, replace with actual paths if needed
        model_config_path = "bewo_config/model_config_sim_mix.json"
        model_ckpt_path = "bewo_config/BEWO_mix.ckpt"

        with open(model_config_path) as f:
            model_config = json.load(f)
        model, model_config = load_model(model_config=model_config, model_ckpt_path=model_ckpt_path, device=args.device)
    
    
    
    if args.manual == True:
        generate_audio_manual(args.prompt, model, args.device, args.init_direction,args.final_direction,args.moving)
    else:
        generate_audio(args.prompt, model, args.device)
    


