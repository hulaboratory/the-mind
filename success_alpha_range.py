
from glob import glob
import json
from tqdm import tqdm

N_GROUPS = 10

llm_results = sorted(glob('results/*/game_*.json'))
all_results_success_by_decile = {}
all_results_time = {}

for result in tqdm(llm_results):
    hand = int(result.split('/')[-1].split('_')[-2])
    model_name = result.split('/')[-1].split('_')[1]
    if '32b' in result:
        continue
    model_file = result.split('/')[-1]
    if 'Bayesian' in result:
        if 'True' in result:
            partner_type = 'bayesian_true'
        else:
            partner_type = 'bayesian_false'
    elif 'Rule' in result:
        if 'True' in result:
            partner_type = 'count_true'
        else:
            partner_type = 'count_false'
    elif 'Random' in result:
        partner_type = 'random'
    elif model_file.count(model_name) == 2:
        print(model_file)
        partner_type = 'llm_same'
        # partner_type = 'llm'
    else:
        partner_type = 'llm_different'
        # continue
        # partner_type = 'llm'

    prompt_type = '_'.join(result.split('/')[1].split('_')[1:])
    if 'millisecond' in prompt_type:
        continue
    if '32B' in model_name:
        prompt_type = '_'.join(result.split('/')[1].split('B')[1:])

    all_results_success_by_decile.setdefault(model_name, {}).setdefault(
        partner_type, {}
    ).setdefault(prompt_type, {})
    all_results_time.setdefault(model_name, {}).setdefault(
        partner_type, {}
    ).setdefault(prompt_type, {})

    data_point = json.load(open(result, 'r'))
    player1 = [x for x, y  in data_point['models'].items() if 'player_0' in x][0]
    player2 = [x for x, y  in data_point['models'].items() if 'player_0' not in x][0]
    actions_in_order = [
        data_point['action'][k]
        for k in sorted(data_point['action'].keys(), key=int)
    ]

    total_actions = len(actions_in_order)
    assert total_actions==110

    success_by_decile = [0] * N_GROUPS
    time_by_decile = [[] for _ in range(N_GROUPS)]
    # time_by_decile =[]
    total_by_decile = [0] * N_GROUPS

    for i, action in enumerate(actions_in_order):
        decile = min(N_GROUPS - 1, (i * N_GROUPS) // total_actions)
        total_by_decile[decile] += 1
        if action['isError'] == False:
            success_by_decile[decile] += 1
            if player1 in action['player_id'] and 'solo_play' not in action:
                if action['observed_alpha'][player2]==None:
                    action['observed_alpha'][player2]=0.5
                # time_by_decile[decile].append(abs(action['observed_alpha'][player1]-action['observed_alpha'][player2]))
                time_by_decile[decile].append(action['observed_alpha'][player1])
                # time_by_decile.append(action['observed_alpha'][player1])
    n_time = sum(x['hesitationTime'] for x in actions_in_order)
    all_results_success_by_decile[model_name][partner_type][prompt_type][hand] = {
        'success_by_decile': success_by_decile,
        'total_by_decile': total_by_decile,
        # 'alpha': time_by_decile
        'alpha': [sum(x)/len(x) if len(x)>0 else None for x in time_by_decile]
    }

    all_results_time[model_name][partner_type][prompt_type][hand] = n_time


with open('analysis_results/time_overall.json', 'w') as f:
    json.dump(all_results_time, f, indent=4)

with open('analysis_results/success_by_decile.json', 'w') as f:
    json.dump(all_results_success_by_decile, f, indent=4)


import json

N_GROUPS = 10

HUMAN_AGENT_FILE = "results/cua_raw_data/the-mind-human-agent.json"
ID_to_exclude = ['need to be filled']

with open(HUMAN_AGENT_FILE, "r") as ha:
    human_agent_result = json.load(ha)

success_by_decile_info = {}

for player_id, play_info in human_agent_result.items():
    if player_id in ID_to_exclude:
        continue

    participant = play_info["partnerType"]
    hand_id = play_info["handId"] if "handId" in play_info else 1
    calibration = "true" if play_info["calibration"] else "false"
    round_data = play_info["roundData"]
    all_levels = []
    if "bayesian" in participant:
        partner_type = "bayesian_" + calibration
    elif "rule" in participant:
        partner_type = "count_" + calibration
    else:
        partner_type = "random"

    key = f"{player_id}_{hand_id}"

    valid_cards = []
    for card in round_data:
        level = card["level"]
        valid_cards.append(card)


    success_by_decile = [0] * N_GROUPS
    total_by_decile = [0] * N_GROUPS
    time_by_decile = [[] for _ in range(N_GROUPS)]
    # time_by_decile=[]
    for i, card in enumerate(valid_cards):
        decile = min(N_GROUPS - 1, (i * N_GROUPS) // total_actions)
        total_by_decile[decile] += 1
        if card['playerId']!='human':
            if card["isError"] == False and "autoPlayed" not in card and "handsSnapshot" in card and len(list(card["handsSnapshot"].keys()))==2:
                alpha_player = card['alphaObs']
        if card["isError"] == False and "autoPlayed" not in card:
            success_by_decile[decile] += 1
            if card['playerId']=='human' and "handsSnapshot" in card and len(list(card["handsSnapshot"].keys()))==2:
                time_by_decile[decile].append(card["alphaObs"])
                # time_by_decile[decile].append(abs(card["alphaObs"]-alpha_player))
                # time_by_decile.append(card["alphaObs"])

    success_by_decile_info.setdefault(partner_type, {})[key] = {
        "success_by_decile": success_by_decile,
        "total_by_decile": total_by_decile,
        'alpha': [sum(x) / len(x) if len(x) > 0 else 0.5 for x in time_by_decile]
        # 'alpha': time_by_decile
    }

with open("analysis_results/human_success_by_decile.json", "w") as f:
    json.dump(success_by_decile_info, f, indent=4)