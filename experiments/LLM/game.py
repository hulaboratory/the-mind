import random
import json
import os
import glob
from agent import Agent
from fixed_hands import FIXED_HANDS
class TheMindGame:
    def __init__(self, players, result_path, hand, time_unit='second'):
        self.players = players #{player_id: player_model}
        self.current_level = 1
        self.hands = {player_id:[] for player_id, model in self.players.items()}
        self.pile = []
        if time_unit=='second':
            self.wait_unit = 1
        elif time_unit =='millisecond':
            self.wait_unit = 1/1000
        elif time_unit=='q_second':
            self.wait_unit = 0.25
        elif time_unit == 'half_second':
            self.wait_unit = 0.5
        else:
            raise ValueError("The time unit isn't supported!")
        self.result_path = result_path
        self.hand = hand
        self.cards = list(range(1, 101))
        self.errors = 0
        self.level_wait=0
        self.card_played=[]
        self.card_played_info={
                'models': {x: str(y) for x, y in players.items()},
                'cards_played':{},
                'acc': 0,
                'time_passed_last_play':0,
                'level': 1,
                'level_wait': 0,
                'action_num':0,
                'decisions':[],
                'current_hand':{player_id:[] for player_id, model in self.players.items()},
                'action': {},
                'alpha': {player_id:self.players[player_id].alpha for player_id, _ in self.players.items()},
                'errors_feedback': None,
                'correct_feedback': None,
            'prompt':'',
            'reasoning': ''
        }

    def start_game(self):
        self.pile = []
        self.card_played = []
        cards = FIXED_HANDS[self.hand][self.current_level]
        for i, (player_id, _) in enumerate(self.players.items()):
            self.hands[player_id] = cards[i].copy()
            self.card_played_info["current_hand"][player_id] = (
                self.hands[player_id].copy()
            )
            print(
                f"Player {player_id} hand: "
                f"{self.hands[player_id]}"
            )

    def get_observation(self):
        return self.card_played_info

    def collect_decisions(self):
        decisions = {player_name: player_model.decide(self.card_played_info) for player_name, player_model in self.players.items()}
        played_decisions = []
        played=0
        for i, (player_id, decision) in enumerate(decisions.items()):
            if decision == "PLAY":
                if len(self.hands[player_id])==0:
                    print(f"Player {player_id} has no cards left to play. Invalid PLAY decision!")
                    played_decisions.append((player_id, "WAIT"))
                else:
                    played+=1
                    played_decisions.append((player_id,("PLAY", min(self.hands[player_id]))))
            elif decision=='WAIT':

                played_decisions.append((player_id, "WAIT"))
        random.shuffle(played_decisions) #randomly shuffle the order in case there are two PLAYs.
        return played_decisions

    def resolve_actions(self):
        decisions = self.collect_decisions()
        decisions_two = [x[1] for x in decisions]
        this_round_pile = []

        for player_id, decision in decisions:

            if isinstance(decision, tuple) and decision[0] == "PLAY":
                the_other_id = [x[0] for x in decisions if x[0] != player_id][0]


                if len(self.hands[player_id])==0:
                    # self.card_played_info['decisions'].append((player_id, 'WAIT'))
                    continue
                card_to_play = min(self.hands[player_id])
                self.card_played_info['action_num'] += 1

                self.card_played.append(card_to_play)
                self.pile.append(card_to_play)
                self.hands[player_id].remove(card_to_play)
                print(f"{decision[1]} played by {player_id}")
                self.card_played_info['cards_played'][self.current_level] = self.card_played
                self.card_played_info['current_hand'][player_id]= list(self.hands[player_id])
                self.card_played_info['decisions'].append((player_id, card_to_play))
                left_cards = self.card_played_info['current_hand'][player_id]+self.card_played_info['current_hand'][the_other_id]
                is_error = len(left_cards)>0 and card_to_play > min(left_cards)

                self.card_played_info['action'][self.card_played_info['action_num']]={
                    'level': self.current_level,

                    'player_id': player_id,
                    'value': card_to_play,
                    'left_cards': {pid: list(self.hands[pid]) for pid in self.players},
                    'isError': is_error,
                    'hesitationTime': self.card_played_info['time_passed_last_play'],
                    'level_wait': self.level_wait,
                    'topBefore': max(self.card_played[:-1]) if len(self.card_played) > 1 else 0,
                    'prompt': self.players[player_id].get_prompt() if isinstance(self.players[player_id], Agent) else ''}

                self.card_played_info['time_passed_last_play'] = 0
                this_round_pile.append(card_to_play)
                the_other_player = self.players[the_other_id]
                partner_hand = self.card_played_info['current_hand'][the_other_id]
                current_player = self.players[player_id]


                if partner_hand:
                    self.players[player_id].observed_alpha = self.players[player_id].estimate_alpha(
                        self.card_played_info)
                    if 'Bayesian' in str(current_player):
                        pass
                    else:
                        self.players[player_id].alpha = max(0.05, self.players[player_id].estimate_alpha(
                            self.card_played_info))

                else:
                    self.card_played_info['action'][self.card_played_info['action_num']]['solo_play'] = True

                if the_other_player.calibration:
                    the_other_player.calibrate(self.card_played_info)


                self.card_played_info['action'][self.card_played_info['action_num']]['alpha'] = \
                        {player_id: self.players[player_id].alpha for player_id, _ in self.players.items()}
                self.card_played_info['action'][self.card_played_info['action_num']]['observed_alpha'] = \
                    {player_id: self.players[player_id].observed_alpha for player_id, _ in self.players.items()
                     }
                # print(self.card_played_info['action_num'])
                # print(self.card_played_info['action'][self.card_played_info['action_num']])
                if is_error:
                    self.errors += 1
                    cards_to_play_automatically = [c for c in left_cards if c<card_to_play]
                    for c in cards_to_play_automatically:
                        for pid in self.players:
                            if c in self.hands[pid]:
                                self.card_played_info['decisions'].append((f'{pid}(autoplay)', c))
                                self.card_played.append(c)
                                self.pile.append(c)

                                self.hands[pid].remove(c)
                                self.card_played_info['current_hand'][pid] = list(self.hands[pid])
                                self.card_played_info['action_num'] += 1
                                self.card_played_info['cards_played'][self.current_level] = self.card_played
                                self.card_played_info['decisions'].append((pid, c))
                                self.card_played_info['action'][self.card_played_info['action_num']] = {
                                    'level': self.current_level,

                                    'player_id': pid,
                                    'value': c,
                                    'left_cards': {pid: list(self.hands[pid]) for pid in self.players},
                                    'isError': 'AutoPlay',
                                    'hesitationTime': 0,
                                    'level_wait': self.level_wait,
                                    'topBefore': max(self.card_played[:-1]) if len(self.card_played) > 1 else 0,
                                    'prompt': self.players[pid].get_prompt() if isinstance(self.players[pid], Agent) else '',
                                    'alpha': {playerid: player.alpha for playerid, player in self.players.items()},
                                    'observed_alpha':{playerid: player.observed_alpha for playerid, player in self.players.items()}
                                }

            elif decision == "WAIT":
                # print("WAIT by", player_id)
                self.card_played_info['decisions'].append((player_id, "WAIT"))
            else:
                print(f"Invalid decision {decision!r} from {player_id}, treating as WAIT")
                self.card_played_info['decisions'].append((player_id, "WAIT"))
        if all(d == "WAIT" for d in decisions_two):
            self.card_played_info['time_passed_last_play'] += self.wait_unit
            self.level_wait += self.wait_unit
            # print(self.card_played_info['time_passed_last_play'])

        os.makedirs(self.result_path, exist_ok=True)
        path = f"{self.result_path}/tmp.json"
        with open(path, 'w') as j:
            json.dump(self.card_played_info, j, indent=2)

    def next_level(self):
        all_empty = all(len(h) == 0 for h in self.hands.values())
        if all_empty:
            if self.current_level >= 10:
                print("Congratulations!")
                return True
            else:
                self.current_level += 1
                self.card_played_info['level']+=1
                self.level_wait = 0
                self.start_game()
                print(f"==========Level {self.current_level} starts now!========")
                return False
        return False

    def game_play(self):
        self.start_game()
        while True:
            self.resolve_actions()
            if self.next_level():
                os.makedirs(self.result_path, exist_ok=True)
                player_info = '_'.join([str(x).split('/')[-1] for x in list(self.card_played_info['models'].values())])
                prefix = f"game_{player_info}_{self.hand}"
                n = len(glob.glob(f"{self.result_path}/{prefix}_*.json")) + 1
                path = f"{self.result_path}/{prefix}_{self.hand}_{n:03d}.json"
                cards_played = [x for y in list(self.card_played_info['cards_played'].values()) for x in y]
                self.card_played_info['acc'] = (len(cards_played)-self.errors)/len(cards_played)
                with open(path, 'w') as j:
                    json.dump(self.card_played_info, j, indent=2)
                print(f"Game saved to {path}")
                return


