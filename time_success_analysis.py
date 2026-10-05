from glob import glob
import json
from tqdm import tqdm
llm_results = sorted(glob('results/*_time_based_second/game_*.json'))
all_results_success = {}
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
    elif model_file.count(model_name)==2:
        print(model_file)
        partner_type = 'llm_same'
        # partner_type= 'llm'
    else:
        # partner_type = 'llm'
        partner_type = 'llm_different'
        # continue
        # partner_type = 'llm'

    prompt_type ='_'.join(result.split('/')[1].split('_')[1:])
    if 'millisecond' in prompt_type:
        continue
    if '32B' in model_name:
        prompt_type = '_'.join(result.split('/')[1].split('B')[1:])
    # if 'time_based' not in prompt_type or partner_type=='llm':
    #     continue
    all_results_success.setdefault(model_name, {}).setdefault(partner_type, {}).setdefault(prompt_type, {})
    all_results_time.setdefault(model_name, {}).setdefault(partner_type, {}).setdefault(prompt_type, {})
    data_point = json.load(open(result, 'r'))

    player_name = list([x for x in data_point['models'].keys() if '0' in x])[0]
    n_success = sum([1 for i, action in data_point['action'].items() if action['isError']==False])
    n_time = sum([x['hesitationTime'] for _, x in data_point['action'].items()])
    all_results_success[model_name][partner_type][prompt_type][hand]={'success': n_success, 'time': n_time}
    # all_results_success[model_name][partner_type][prompt_type][hand] =n_success
    all_results_time[model_name][partner_type][prompt_type][hand]=n_time




with open('analysis_results/overall_time_llm.json', 'w') as f:
    json.dump(all_results_time, f, indent=4)
#
with open('analysis_results/overall_performance_llm_time_based_second.json', 'w') as f:
    json.dump(all_results_success, f, indent=4)


# with open('analysis_results/time_overall_time_based.json', 'w') as f:
#     json.dump(all_results_time, f, indent=4)
#
# with open('analysis_results/success_overall_time_based.json', 'w') as f:
#     json.dump(all_results_success, f, indent=4)



human_agent_file = 'results/cua_raw_data/the-mind-gpt-luna_updated.json'
ID_to_exclude = ["need to be filled"]
with open(human_agent_file, 'r') as ha:
    human_agent_result = json.load(ha)

alpha_info={}
success_time_info={}
for i, (player_id, play_info) in enumerate(human_agent_result.items()):
    if player_id in ID_to_exclude:
        continue
    participant=play_info['partnerType']
    gamma = play_info['gamma']
    if 'handId' not in play_info:
        hand_id=1
    else:
        hand_id = play_info['handId']
    lr= play_info['lr']
    theta=0.85
    calibration='true' if play_info['calibration'] else 'false'
    play_info = play_info['roundData']

    if 'bayesian' in participant:
        partner_type = 'bayesian'+'_'+calibration
    elif 'rule' in participant:
        partner_type = 'count'+'_'+calibration
    else:
        partner_type = 'random'

    player_id += '_'+str(hand_id)
    alpha_info.setdefault(partner_type, {}).setdefault(player_id, {})
    success_time_info.setdefault(partner_type, {}).setdefault(hand_id, [])


    alpha_info[partner_type][player_id]['alpha']=[]
    alpha_info[partner_type][player_id]['alpha_diff']=[]
    success=0
    time_total = 0
    finished_levels = set()
    for card in play_info:
        time_total += card["hesitationTime"] / 1000
        level = card["level"]
        # if level in finished_levels:
        #     continue
        if card["isError"]==False and 'autoPlayed' not in card:
            success+=1

        if 'autoPlayed' not in card and card["playerId"] == 'agent':
            lastest_alpha = card["alphaObs"]


        if 'autoPlayed' not in card and card["playerId"]=='human':
            # print(card)
            alpha_info[partner_type][player_id]['alpha'].append(card["alphaObs"])

            if 'agentAlpha' in card:
                alpha_info[partner_type][player_id]['alpha_diff'].append(abs(card["alphaObs"]-card["agentAlpha"]))
            else:
                alpha_info[partner_type][player_id]['alpha_diff'].append(abs(card["alphaObs"]-lastest_alpha))

        # if 'handsSnapshot' in card:
        #     snapshot = card["handsSnapshot"]
        #     number_of_players = len(snapshot)
        #     if number_of_players <2:
        #         finished_levels.add(level)

    success_time_info[partner_type][hand_id].append({'success': success, 'hesitation_time': time_total})


with open('analysis_results/gptluna_computer_use_success_time.json', 'w') as f:
    json.dump(success_time_info, f, indent=4)

#
# with open('analysis_results/human_alpha_info.json', 'w') as f:
#     json.dump(alpha_info, f, indent=4)


