from game import TheMindGame
from agent import Agent, RandomPlayer, RuleBasedAgent, BayesianPlayer
from cerebras.cloud.sdk import Cerebras
from transformers import pipeline,AutoModelForCausalLM
import torch
from vllm import LLM, SamplingParams
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse
import os
from multiprocessing import freeze_support


def load_model(model_name, vllm=True, device=None):
    if vllm:
        pipe = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=16384,
            tensor_parallel_size=1,
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
        llm_player = Agent(player_name, 'openai/gpt-oss-120b', model=model, prompt_type=prompt_type,
                           feedback=True,
                           using_API=False)

    elif model_name == 'gpt-oss-20b':
        model_path = "/scratch/jhu/jhu35/xyang180/huggingface-cache/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
        model = load_model(model_path, True)

        if player1:
            player_name = 'player_0_gptoss-20'
        else:
            player_name = 'player_1_gptoss-20'
        llm_player = Agent(player_name, 'openai/gpt-oss-20b', model=model, prompt_type=prompt_type,
                           feedback=True,
                           using_API=False)
    elif model_name == 'qwen3-32b':
        model_path = '/scratch/jhu/jhu35/xyang180/huggingface-cache/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137'
        tokenizer, model = load_model(model_path, False,1)

        if player1:
            player_name = 'player_0_Qwen3-32B'
        else:
            player_name = 'player_1_Qwen3-32B'

        llm_player = Agent(player_name, 'Qwen/Qwen3-32B', transformer=True, tokenizer=tokenizer, model=model, prompt_type=prompt_type,
                           feedback=True,
                           using_API=False)

    elif model_name == 'qwen3-8b':
        model_path = '/scratch/jhu/jhu35/xyang180/huggingface-cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218'
        tokenizer, model = load_model(model_path, False,1)

        if player1:
            player_name = 'player_0_Qwen3-8B'
        else:
            player_name = 'player_1_Qwen3-8B'
        llm_player = Agent(player_name, 'Qwen/Qwen3-8B', transformer=True,tokenizer=tokenizer, model=model, prompt_type=prompt_type,
                           feedback=True,
                           using_API=False)
    elif model_name == 'llama70b':
        RESULT_PATH='results/llama70b'+ '_'+ prompt_type
        model_path = '/scratch/jhu/jhu35/xyang180/huggingface-cache/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b'
        model = load_model(model_path, True)
        if player1:
            player_name = 'player_0_llama70b'
        else:
            player_name = 'player_1_llama70b'

        llm_player = Agent(player_name, 'meta-llama/Llama-3.1-70B-Instruct', model=model, prompt_type=prompt_type,
                            feedback=True,
                            using_API=False)

    elif model_name == 'gemini':

        if player1:
            player_name = 'player_0_gemini'
        else:
            player_name = 'player_1_gemini'


        llm_player = Agent(player_name, "gemini-3.7-flash", prompt_type=prompt_type,
                           feedback=True,
                           using_API=True)
    elif model_name =='gpt-6':
        llm_player= Agent('player_0_gpt-6', "gpt-6-astra", prompt_type=prompt_type, feedback=True, using_API=True)
        if player1:
            player_name = 'player_0_gpt-6'
        else:
            player_name = 'player_1_gpt-6'

    elif model_name == 'gpt-5.6':
        llm_player = Agent('player_0_gpt-5.6', "gpt-5.6-luna", prompt_type=prompt_type, feedback=True, using_API=True)
        if player1:
            player_name = 'player_0_gpt-5.6'
        else:
            player_name = 'player_1_gpt-5.6'

    else:
        raise ValueError(f'{model_name} is not available! Please choose from the following: [gpt-oss-120b, gpt-oss-20b, llama-3.1-8b]')

    return player_name, llm_player

def main():
    parser = argparse.ArgumentParser('LLM based agent play')
    parser.add_argument('model_name1', type=str, help='model name1')
    parser.add_argument('model_name2', type=str, help='model name2')
    parser.add_argument('prompt_type', type=str, help='prompt_type')
    args = parser.parse_args()
    prompt_type = args.prompt_type
    model_name1 = args.model_name1
    model_name2 = args.model_name2
    model_path_1 = model_name1.split('/')[-1]
    model_path_2 = model_name2.split('/')[-1]
    result_path_model1=f'results/{model_path_1}_{prompt_type}'
    os.makedirs(result_path_model1, exist_ok=True)
    result_path_model2 = f'results/{model_path_2}_{prompt_type}'
    os.makedirs(result_path_model2, exist_ok=True)
    player_name1, llm_player1 = load_models(model_name1, prompt_type, True)
    if 'time_based' in prompt_type:
        player_name2, llm_player2 = load_models(model_name2, 'time_based_second', False)
    else:
        player_name2, llm_player2 = load_models(model_name2, prompt_type, False)
    for i in tqdm(range(10,11)):
        players = {
            player_name1: llm_player1,
            player_name2: llm_player2
        }

        game = TheMindGame(players, result_path_model1, hand=i)
        game.game_play()

if __name__ == '__main__':
    freeze_support()
    main()
