
def generate_prompt(cards_played_info, player_id, feedback=True, prompt_type='time_based_second'):
    cards_played = cards_played_info['cards_played']
    level = cards_played_info['level']
    current_hand = cards_played_info['current_hand'][player_id]
    current_played_cards = cards_played[level] if level in cards_played else None

    prompt = "Below, you will see the 5 most recent plays in the game and the current game state. If the 5 most recent plays include the beginning of a new level, it will be clearly marked. Your task will be to decide whether to continue waiting or play your lowest card. \n"

    if feedback:
        memory_feedback = memory_build(cards_played_info, player_id,5, prompt_type)
        prompt+=memory_feedback

    prompt += "\n\n===CURRENT GAME STATE===\n"
    prompt += f"You are currently on level {level}.\n"
    if current_hand:
        smallest_card = sorted(current_hand)[0]
        prompt += f"Current hand: {current_hand}\n"
    else:
        return 'WAIT'

    the_other_num = len([cards_played_info['current_hand'][x] for x in list(cards_played_info['current_hand'].keys()) if x!=player_id][0])

    prompt+=f'Your partner has {the_other_num} card(s) left. \n'
    prompt+=game_basic_info(current_played_cards, prompt_type, cards_played_info)

    prompt+='\n\n===YOUR TASK===\n'

    if prompt_type=='time_based_second':
        prompt+= f'Based on the above information, do you want to wait one more second or play your lowest card, i.e., {smallest_card}? \n'
        prompt += (
            "\nRespond with exactly one word:\n"
            "PLAY\n"
            "or\n"
            "WAIT"
        )
    elif prompt_type=='time_based_half_second':
        prompt += f'Based on the above information, do you want to wait 0.5 second or play your lowest card, i.e., {smallest_card}? \n'
        prompt += (
            "\nRespond with exactly one word:\n"
            "PLAY\n"
            "or\n"
            "WAIT"
        )
    elif prompt_type =='time_based_q_second':
        prompt += f'Based on the above information, do you want to wait another 0.25 second or play your lowest card, i.e., {smallest_card}? \n'
        prompt += (
            "\nRespond with exactly one word:\n"
            "PLAY\n"
            "or\n"
            "WAIT"
        )
    elif prompt_type=='time_based_millisecond':
        prompt += f'Based on the above information, do you want to wait one more millisecond or play your lowest card, i.e., {smallest_card}? \n'
        prompt += (
            "\nRespond with exactly one word:\n"
            "PLAY\n"
            "or\n"
            "WAIT"
        )

    elif prompt_type =='wait_based':
        prompt += f'Based on the above information, do you want to wait one more time or play your lowest card, i.e., {smallest_card}? \n'
        prompt += (
            "\nRespond with exactly one word:\n"
            "PLAY\n"
            "or\n"
            "WAIT"
        )
    elif prompt_type =='wait_count_based':
        prompt += f'Based on the above information, do you want to wait one more time or play your lowest card, i.e., {smallest_card}? \n'
        prompt += (
            "\nRespond with exactly one word:\n"
            "PLAY\n"
            "or\n"
            "WAIT"
        )

    else:
        prompt += f"\nBased on the information above, how many seconds would you like to wait before playing your lowest card, i.e., {smallest_card}?\n"
        prompt += "Respond with only a single numeric value. Do not include any explanation or units.\n"
    print('===========================================')
    print(player_id)
    print(prompt)
    print('===========================================')
    return prompt

def play_judge(error_key):
    if error_key==True:
        return 'Error'
    elif error_key=='AutoPlay':
        return 'This card is automatically played because of the previous error made.'
    else:
        return 'Success'

def identify_solo_dump(card, prompt_type, level=True):
    if level:
        level_wait = card['level_wait']
    else:
        level_wait = card['hesitationTime']

    unit = 'second'

    if card['hesitationTime']==0 and 'solo_play' in card:
        if prompt_type=='wait_count_based':
            return f'{level_wait} (This card is played immediately because it is the last remaining card in this level).'
        elif prompt_type=='wait_based':
            return 'This card is played immediately because it is the last remaining card in this level.'
        elif 'time_based' in prompt_type:
            level_wait = round(level_wait, 5)
            return f'{level_wait} {unit} (This card is played immediately because it is the last remaining card in this level).'
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
    else:
        if prompt_type=='wait_based':
            if level_wait==0:
                return 'PLAY'
            else:
                return ', '.join(int(level_wait)*['WAIT'])+', PLAY'
        elif prompt_type=='wait_count_based':
            return level_wait
        else:
            level_wait = round(level_wait, 4)
            return f'{level_wait} {unit}(s).'

def game_basic_info(current_played_cards,  prompt_type, cards_played_info):


    unit = 'second'

    info =''

    if current_played_cards:
        highest_card = max(current_played_cards)
        end = f'since the last successfully played card {highest_card}'
    else:
        current_l = cards_played_info['level']
        end = f'since the start of level {current_l}'

    if current_played_cards:
        highest_card = max(current_played_cards)
        # prompt += f"Cards played this level: {current_played_cards}\n"
        info += f"The highest card played so far: {highest_card}\n"
        if prompt_type=='wait_count_based':

            info += f"You have chosen to WAIT {cards_played_info['time_passed_last_play']} times {end}.\n"
        elif prompt_type=='wait_based':
            if int(cards_played_info['time_passed_last_play'])==0:
                info+='You have not made any decision yet.'
            else:
                waits=','.join(int(cards_played_info['time_passed_last_play'])*['WAIT'])
                info += f"Your decision(s) {end} are: {waits}.\n"
        elif 'time_based' in prompt_type:
            wait_time = round(cards_played_info['time_passed_last_play'], 4)
            info += f"You have been waiting for {wait_time} {unit}(s) {end}.\n"
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
    else:
        info += f"No cards have been played yet.\n"
        if cards_played_info['time_passed_last_play']==0:
            info += f"You have not made any decisions {end}.\n"
        else:
            if prompt_type == 'wait_count_based':
                info += f"You have made {cards_played_info['time_passed_last_play']} WAIT decisions in total {end}.\n"
            elif prompt_type == 'wait_based':
                if int(cards_played_info['time_passed_last_play']) == 0:
                    info += f'You have not made any decision {end}.'
                else:
                    waits = ','.join(int(cards_played_info['time_passed_last_play']) * ['WAIT'])
                    info += f"Your decisions {end} are: {waits}.\n"
            elif 'time_based' in prompt_type:
                wait_time = round(cards_played_info['time_passed_last_play'], 4)
                info += f"You have been waiting for {wait_time} {unit}(s) {end}.\n"
            else:
                raise ValueError(f"Unknown prompt type: {prompt_type}")
    return info

def memory_build(cards_played_info, player_id,chunk, prompt_type):

    if cards_played_info['action']:
        actions = list(cards_played_info['action'].items())
        if prompt_type=='wait_count_based':
            memory = [
                    (f"-----BEGIN LEVEL {card['level']}, DEAL NEW HANDS AND RESET CLOCK-----\n"
                     + f"Level: {card['level']}\n"
                       f"Card played: {card['value']}\n"
                       f"Player: {'You' if card['player_id'] == player_id else 'Your partner'}\n"
                       f"Number of WAITs since the start of level {card['level']}: {identify_solo_dump(card, prompt_type)}\n"
                       f"Result: {play_judge(card['isError'])}"
                    if i==0 or card['level']!=actions[i-1][1]['level']
                    else
                     f"Level: {card['level']}\n"
                    f"Card played: {card['value']}\n"
                    f"Player: {'You' if card['player_id'] == player_id else 'Your partner'}\n"
                    f"Number of WAITs since the last played card {actions[i-1][1]['value']}: {identify_solo_dump(card, prompt_type, level=False)}\n"
                    f"Result: {play_judge(card['isError'])}")

                for i, (action, card) in enumerate(actions)
            ][-chunk:]
        elif prompt_type=='wait_based':
            memory = [
                ((f"-----BEGIN LEVEL {card['level']}, DEAL NEW HANDS AND RESET CLOCK-----\n"
                  f"Level: {card['level']}\n"
                  f"Card played: {card['value']}\n"
                  f"Player: {'You' if card['player_id'] == player_id else 'Your partner'}\n"
                  f"Decisions since the start of level {card['level']}: {identify_solo_dump(card, prompt_type)}\n"
                  f"Result: {play_judge(card['isError'])}"
                    if i==0 or card['level']!=actions[i-1][1]['level'] or len(actions)==1
                    else
                  f"Level: {card['level']}\n"
                  f"Card played: {card['value']}\n"
                  f"Player: {'You' if card['player_id'] == player_id else 'Your partner'}\n"
                  f"Decisions since the last played card {actions[i-1][1]['value']}: {identify_solo_dump(card, prompt_type, level=False)}\n"
                  f"Result: {play_judge(card['isError'])}"
                  )
                )
             for i, (action, card) in enumerate(actions)
            ][-chunk:]
        elif 'time_based' in prompt_type:
            memory = [
                ( (f"-----BEGIN LEVEL {card['level']}, DEAL NEW HANDS AND RESET CLOCK-----\n"
                   f"Level: {card['level']}\n"
                   f"Card played: {card['value']}\n"
                   f"Player: {'You' if card['player_id'] == player_id else 'Your partner'}\n"
                   f"Time since the start of level {card['level']}: {identify_solo_dump(card, prompt_type)}\n"
                   f"Result: {play_judge(card['isError'])}"
                    if i==0 or card['level']!=actions[i-1][1]['level'] or len(actions)==1
                    else
                    f"Level: {card['level']}\n"
                    f"Card played: {card['value']}\n"
                    f"Player: {'You' if card['player_id'] == player_id else 'Your partner'}\n"
                    f"Time since the last played card {actions[i-1][1]['value']}: {identify_solo_dump(card, prompt_type, level=False)}\n"
                    f"Result: {play_judge(card['isError'])}"
                ))
                for i, (action, card) in enumerate(actions)
            ][-chunk:]
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
    else:
        return "\n\n===5 MOST RECENT PLAYS===\n" + 'No plays have been made yet.'
    return "\n\n===5 MOST RECENT PLAYS===\n" + "\n\n".join(memory)+'\n\n'


