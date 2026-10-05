from game import TheMindGame
from agent import Agent, RandomPlayer, RuleBasedAgent, BayesianPlayer
from transformers import pipeline,AutoModelForCausalLM
import torch
from vllm import LLM, SamplingParams
from tqdm import tqdm
import argparse
import os
from multiprocessing import freeze_support


def load_model(model_name, gpu=1, vllm=True):
    if vllm:
        if 'llama' in model_name.lower():
            pipe = LLM(
                model=model_name,
                tensor_parallel_size=gpu,
                trust_remote_code=True,
                max_model_len=16384,)
        else:
            pipe = LLM(
                model=model_name,
                tensor_parallel_size=gpu,
                trust_remote_code=True,
            )

    else:
        pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    return pipe



def load_models(args, gpu, prompt_type):
    if args.model_name == 'gpt-oss-120b':
        RESULT_PATH='results/gpt-oss-120b'+'_'+ prompt_type
        model_path = "openai/gpt-oss-120b"
        model = load_model(model_path, gpu, True)
        llm_player = Agent('player_0_gptoss-120', 'openai/gpt-oss-120b', model=model, prompt_type=prompt_type,
                           feedback=True,
                           using_API=False)
        llm_player1 = Agent('player_1_gptoss-120', 'openai/gpt-oss-120b', model=model, prompt_type=prompt_type,
                            feedback=True,
                            using_API=False)

        player_name = 'player_0_gptoss-120'
        player_name1 = 'player_1_gptoss-120'
    elif args.model_name == 'gpt-oss-20b':
        RESULT_PATH='results/gpt-oss-20b'+'_'+ prompt_type
        model_path =  'openai/gpt-oss-20b'

        model = load_model(model_path, gpu, True)
        llm_player = Agent('player_0_gptoss-20', 'openai/gpt-oss-20b', model=model, prompt_type=prompt_type,
                           feedback=True,
                           using_API=False)
        llm_player1 = Agent('player_1_gptoss-20', 'openai/gpt-oss-20b', model=model, prompt_type=prompt_type,
                            feedback=True,
                            using_API=False)

        player_name = 'player_0_gptoss-20'
        player_name1 = 'player_1_gptoss-20'

    elif args.model_name == 'mistral-medium':
        RESULT_PATH = 'results/mistral-medium' + '_' + prompt_type
        model_path = 'mistralai/Mistral-Medium-3.5-128B'

        model = load_model(model_path, gpu, True)
        llm_player = Agent('player_0_mistralm', 'mistralai/Mistral-Medium-3.5-128B-EAGLE', model=model, prompt_type=prompt_type,
                           feedback=True,
                           using_API=False)
        llm_player1 = Agent('player_1_mistralm', 'mistralai/Mistral-Medium-3.5-128B-EAGLE', model=model, prompt_type=prompt_type,
                            feedback=True,
                            using_API=False)

        player_name = 'player_0_mistralm'
        player_name1 = 'player_1_mistralm'

    elif args.model_name == 'qwen3-32b':
        RESULT_PATH='results/Qwen3-32B'+prompt_type
        model_path = 'Qwen/Qwen3-32B'

        model = load_model(model_path, gpu, True)
        llm_player = Agent('player_0_Qwen3-32B', 'Qwen/Qwen3-32B', model=model, prompt_type=prompt_type,
                           feedback=True,
                           using_API=False)
        llm_player1 = Agent('player_1_Qwen3-32B', 'Qwen/Qwen3-32B', model=model, prompt_type=prompt_type,
                            feedback=True,
                            using_API=False)

        player_name = 'player_0_Qwen3-32B'
        player_name1 = 'player_1_Qwen3-32B'

    elif args.model_name == 'llama8b':
        RESULT_PATH='results/llama8b'+ '_'+ prompt_type
        model_path = 'meta-llama/Llama-3.1-8B-Instruct'

        model = load_model(model_path, gpu, True)
        llm_player = Agent('player_0_llama8b', 'meta-llama/Llama-3.1-8B-Instruct', model=model, prompt_type=prompt_type,
                           feedback=True,
                           using_API=False)
        llm_player1 = Agent('player_1_llama8b', 'meta-llama/Llama-3.1-8B-Instruct', model=model, prompt_type=prompt_type,
                            feedback=True,
                            using_API=False)

        player_name = 'player_0_llama8b'
        player_name1 = 'player_1_llama8b'

    elif args.model_name == 'llama1b':
        RESULT_PATH='results/llama1b'+ '_'+ prompt_type
        model_path = 'meta-llama/Llama-3.2-1B-Instruct'

        model = load_model(model_path, gpu, True)
        llm_player = Agent('player_0_llama1b', 'meta-llama/Llama-3.2-1B-Instruct', model=model, prompt_type=prompt_type,
                           feedback=True,
                           using_API=False)
        llm_player1 = Agent('player_1_llama1b', 'meta-llama/Llama-3.2-1B-Instruct', model=model, prompt_type=prompt_type,
                            feedback=True,
                            using_API=False)

        player_name = 'player_0_llama1b'
        player_name1 = 'player_1_llama1b'

    elif args.model_name == 'llama70b':
        RESULT_PATH='results/llama70b'+ '_'+ prompt_type
        model_path = 'meta-llama/Llama-3.1-70B-Instruct'

        model = load_model(model_path, gpu, True)
        llm_player = Agent('player_0_llama70b', 'meta-llama/Llama-3.1-70B-Instruct', model=model, prompt_type=prompt_type,
                           feedback=True,
                           using_API=False)
        llm_player1 = Agent('player_1_llama70b', 'meta-llama/Llama-3.1-70B-Instruct', model=model, prompt_type=prompt_type,
                            feedback=True,
                            using_API=False)

        player_name = 'player_0_llama70b'
        player_name1 = 'player_1_llama70b'

    elif args.model_name == 'qwen3-8b':
        RESULT_PATH='results/Qwen3-8B'+ '_'+ prompt_type
        model_path = 'Qwen/Qwen3-8B'

        model = load_model(model_path, gpu, True)
        llm_player = Agent('player_0_Qwen3-8B', 'Qwen/Qwen3-8B', model=model, prompt_type=prompt_type,
                           feedback=True,
                           using_API=False)
        llm_player1 = Agent('player_1_Qwen3-8B', 'Qwen/Qwen3-8B', model=model, prompt_type=prompt_type,
                            feedback=True,
                            using_API=False)

        player_name = 'player_0_Qwen3-8B'
        player_name1 = 'player_1_Qwen3-8B'

    elif args.model_name =='gemini':
        RESULT_PATH= 'results/gemini3.7_'+prompt_type
        llm_player= Agent('player_0_gemini', "gemini-3.7-flash", prompt_type=prompt_type, feedback=True, using_API=True)
        llm_player1 = Agent('player_1_gemini', "gemini-3.7-flash", prompt_type=prompt_type, feedback=True, using_API=True)
        player_name = 'player_0_gemini'
        player_name1 = 'player_1_gemini'

    elif args.model_name =='gpt-5.6-luna':
        RESULT_PATH= 'results/gpt-5.6-luna_'+prompt_type
        llm_player= Agent('player_0_gpt-5.6-luna', "gpt-5.6-luna", prompt_type=prompt_type, feedback=True, using_API=True)
        llm_player1 = Agent('player_1_gpt-5.6-luna', "gpt-5.6-luna", prompt_type=prompt_type, feedback=True, using_API=True)
        player_name = 'player_0_gpt-5.6-luna'
        player_name1 = 'player_1_gpt-5.6-luna'

    elif args.model_name =='gpt-6':
        RESULT_PATH= 'results/gpt-6_'+prompt_type
        llm_player= Agent('player_0_gpt-6', "gpt-6-astra", prompt_type=prompt_type, feedback=True, using_API=True)
        llm_player1 = Agent('player_1_gpt-6', "gpt-6-astra", prompt_type=prompt_type, feedback=True, using_API=True)
        player_name = 'player_0_gpt-6'
        player_name1 = 'player_1_gpt-6'

    else:
        raise ValueError(f'{args.model_name} is not available! Please choose from the following: [gpt-oss-120b, gpt-oss-20b, llama-3.1-8b]')
    os.makedirs(RESULT_PATH, exist_ok=True)

    return player_name, player_name1, llm_player, llm_player1, RESULT_PATH

def main():
    parser = argparse.ArgumentParser('LLM based agent play')
    parser.add_argument('model_name', type=str, help='model name')
    parser.add_argument('prompt_type', type=str, help='prompt_type')
    parser.add_argument('gpu', type=int, help='number of gpus available')
    args = parser.parse_args()
    gpu = args.gpu
    prompt_type = args.prompt_type
    if 'half_second' in prompt_type:
        time_unit='half_second'
    elif 'millisecond' in prompt_type:
        time_unit = 'millisecond'
    elif 'q_second' in prompt_type:
        time_unit = 'q_second'
    else:
        time_unit = 'second'
    player_name, player_name1, llm_player, llm_player1, RESULT_PATH = load_models(args, gpu, prompt_type)

    for i in tqdm(range(1,11)):
        players = {
            player_name: llm_player,
            'rule_based_player': RuleBasedAgent('rule_based_player', alpha=0.5, lr=0.3, calibration=True),
        }

        game = TheMindGame(players, RESULT_PATH, time_unit=time_unit, hand=i)
        game.game_play()

        players = {
            player_name: llm_player,
            'rule_based_player': RuleBasedAgent('rule_based_player', alpha=0.5, lr=0.3, calibration=False),
        }

        game = TheMindGame(players, RESULT_PATH,time_unit=time_unit, hand=i)
        game.game_play()

        players = {
             player_name: llm_player,
            'bayesian_based_player': BayesianPlayer('bayesian_based_player', alpha=0.5, lr=0.3, gamma=2, theta=0.85, calibration=True),
        }

        game = TheMindGame(players, RESULT_PATH, time_unit=time_unit,hand=i)
        game.game_play()

        players = {
            player_name: llm_player,
            'bayesian_based_player': BayesianPlayer('bayesian_based_player', alpha=0.5, lr=0.3, gamma=2, theta=0.85,
                                                    calibration=False),
        }

        game = TheMindGame(players, RESULT_PATH,time_unit=time_unit, hand=i)
        game.game_play()

        players = {
            player_name: llm_player,
            'random_player': RandomPlayer('random_player'),
        }

        game = TheMindGame(players, RESULT_PATH, time_unit=time_unit,hand=i)
        game.game_play()

        players = {
            player_name: llm_player,
            player_name1: llm_player1
        }

        game = TheMindGame(players, RESULT_PATH, time_unit=time_unit, hand=i)
        game.game_play()


if __name__ == '__main__':
    freeze_support()
    main()
