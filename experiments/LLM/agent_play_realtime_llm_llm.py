from game_realtime import TheMindGame
from agent import BayesianPlayerTime, RuleBasedAgentTime, RandomPlayerTime, AgentTime
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from vllm import LLM, SamplingParams
from tqdm import tqdm
import argparse
import os
from multiprocessing import freeze_support

def load_model(model_name, vllm=True, device=None):
    if vllm:
        pipe = LLM(
            model=model_name,
            trust_remote_code=True,
        )
        return pipe
    else:

        tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": device},

        )
        return tokenizer, model




def load_models(model_name, prompt_type, player1=True):
    if model_name == 'gpt-oss-120b':
        model_path = "/scratch/jhu/jhu35/xyang180/huggingface-cache/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
        model = load_model(model_path, True)

        if player1:
            player_name = 'player_0_gptoss-120'
        else:
            player_name = 'player_1_gptoss-120'
        llm_player = AgentTime(player_name, 'openai/gpt-oss-120b', model=model, prompt_type=prompt_type,
                           feedback=True,
                           using_API=False)

    elif model_name == 'gpt-oss-20b':
        model_path = "/scratch/jhu/jhu35/xyang180/huggingface-cache/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
        model = load_model(model_path, True)

        if player1:
            player_name = 'player_0_gptoss-20'
        else:
            player_name = 'player_1_gptoss-20'
        llm_player = AgentTime(player_name, 'openai/gpt-oss-20b', model=model, prompt_type=prompt_type,
                           feedback=True,
                           using_API=False)
    elif model_name == 'qwen3-32b':
        model_path = '/scratch/jhu/jhu35/xyang180/huggingface-cache/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137'
        tokenizer, model = load_model(model_path, False,1)

        if player1:
            player_name = 'player_0_Qwen3-32B'
        else:
            player_name = 'player_1_Qwen3-32B'

        llm_player = AgentTime(player_name, 'Qwen/Qwen3-32B', tokenizer=tokenizer, model=model, prompt_type=prompt_type,
                           feedback=True,
                           using_API=False)

    elif model_name == 'qwen3-8b':
        model_path = '/scratch/jhu/jhu35/xyang180/huggingface-cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218'
        tokenizer, model = load_model(model_path, False,1)

        if player1:
            player_name = 'player_0_Qwen3-8B'
        else:
            player_name = 'player_1_Qwen3-8B'
        llm_player = AgentTime(player_name, 'Qwen/Qwen3-8B', tokenizer=tokenizer, model=model, prompt_type=prompt_type,
                           feedback=True,
                           using_API=False)

    elif model_name == 'gemini':

        if player1:
            player_name = 'player_0_gemini'
        else:
            player_name = 'player_1_gemini'


        llm_player = AgentTime(player_name, "gemini-3.7-flash", prompt_type=prompt_type,
                           feedback=True,
                           using_API=True)
    else:
        raise ValueError(f'{model_name} is not available! Please choose from the following: [gpt-oss-120b, gpt-oss-20b, llama-3.1-8b]')

    return player_name, llm_player

def main():
    parser = argparse.ArgumentParser('LLM based agent play')
    parser.add_argument('model_name1', type=str, help='model name1')
    parser.add_argument('model_name2', type=str, help='model name2')
    args = parser.parse_args()

    prompt_type = 'time_based2'
    model_name1 = args.model_name1
    model_name2 = args.model_name2
    result_path_model1=f'{model_name1}_{prompt_type}'
    os.makedirs(result_path_model1, exist_ok=True)
    result_path_model2 = f'{model_name2}_{prompt_type}'
    os.makedirs(result_path_model2, exist_ok=True)
    player_name1, llm_player1 = load_models(model_name1, prompt_type, True)
    player_name2, llm_player2 = load_models(model_name2, prompt_type, False)


    for i in tqdm(range(1,11)):
        players = {
            player_name1: llm_player1,
            player_name2: llm_player2
        }

        game = TheMindGame(players, result_path_model1, hand=i)
        game.game_play()


if __name__ == '__main__':
    freeze_support()
    main()
