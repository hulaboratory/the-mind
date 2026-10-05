from LLM_prompt import generate_prompt
from cerebras.cloud.sdk import Cerebras
import torch
from transformers import AutoTokenizer
import random
from math import comb
import math
from google import genai
from google.genai import types
import json
from vllm import LLM, SamplingParams
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
)
from openai import OpenAI

OPENAI_API_KEY=''
ZAI_API = ''
GEMINI_API=''


def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

class Agent:
    def __init__(self, agent_id, model_name, feedback, prompt_type, tokenizer=None,transformer=False, model=None, calibration=False, using_API=True):
        self.agent_id = agent_id
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.feedback = feedback
        self.transformer=transformer
        self.through_API = using_API
        self.calibration = calibration
        self.prompt_type = prompt_type
        self.alpha = None
        self.observed_alpha = self.alpha
        self.system_prompt = (
            """
            You and your partner are playing a cooperative card game. This game is similar to a game called The Mind. However, our rules are slightly different, so please read the instructions very carefully.

            You and your partner play together as one team. 
            At each round (called a level), you and your partner are each dealt a new hand from a fresh deck of cards numbered 1 to 100. Cards from previous levels do not carry over or affect the cards dealt in the new level. You receive one card each in level 1, two cards each in level 2, and so on.

            Your shared goal is to arrange your and your partner's cards in ascending order, from lowest to highest. You cannot see your partner’s cards. You only know how many cards they have left and what cards have been played.

            How it works: 
            1. Each level deals you a new hand. In level N, you and your partner each hold N cards.
            2. You can only play your lowest card at any time. When you believe it is lower than every card your partner still holds, play it. There are no turns — whenever you sense your lowest card should go next, you play it.
            3. If you play a card while your partner still holds a lower one, that is an error — you played too early. When this happens the game pauses automatically, and every card lower than the one you just played is revealed and removed from your partner's hand. 
            4. Clear all cards to finish the level and move on. There are 10 levels in total.

            The performance of the game is equally based on **how fast you finish** and **how few errors you make**. 
            In other words, you should play QUICKLY and make as FEW errors as possible.
            \n
            """
        )

        self.last_prompt = ''
        if self.through_API:
            if 'zai' in model_name:
                self.client = Cerebras(api_key=ZAI_API)
            elif 'gemini' in model_name:
                self.client = genai.Client(api_key=GEMINI_API,
                                           http_options={'timeout': 300000})
            elif 'gpt' in model_name.lower():
                self.client = OpenAI()
        else:
            if 'oss' in model_name:
                self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
                stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()
                self.sampling = SamplingParams(
                    max_tokens=10000,
                    temperature=0.7,
                    stop_token_ids=stop_token_ids,
                )
            elif 'qwen' in model_name.lower() and transformer==False:
                self.sampling = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=10000)
            elif 'mistral' in model_name.lower():
                self.sampling = SamplingParams(temperature=0.7, max_tokens=10000)
            elif 'llama' in model_name.lower():
                self.sampling = SamplingParams(temperature=0.6, max_tokens=10000)
            self.model = model

    def __str__(self):
        return self.model_name

    def prompt_construct(self, cards_played_info):
        prompt = generate_prompt(cards_played_info, self.agent_id, self.feedback, self.prompt_type)
        self.last_prompt = prompt
        return prompt

    def get_prompt(self):
        return self.last_prompt

    def estimate_alpha(self, cards_played_info):
        action = cards_played_info['action_num']
        if cards_played_info['action'][action]['topBefore']:
            return cards_played_info['action'][action]['hesitationTime'] / (
                        cards_played_info['action'][action]['value'] - cards_played_info['action'][action]['topBefore'])
        else:
            return cards_played_info['action'][action]['hesitationTime'] / (
            cards_played_info['action'][action]['value'])

    def decide(self, cards_played_info):
        prompt = self.prompt_construct(cards_played_info)
        if prompt == 'WAIT':
            return 'WAIT'

        max_retries = 5
        messages = [
            {"role": "system", "content": f"{self.system_prompt}"},
            {"role": "user", "content": f"{prompt}"},
        ]

        partner_id = next(
            pid for pid in cards_played_info['current_hand']
            if pid != self.agent_id
        )

        partner_hand = cards_played_info['current_hand'][partner_id]
        if not partner_hand:
            return 'PLAY'
        current_hand = cards_played_info['current_hand'][self.agent_id]
        if not current_hand:
            return 'WAIT'

        for _ in range(max_retries):
            if self.through_API:
                if 'zai' in self.model_name:

                    response = self.client.chat.completions.create(
                        messages=messages,
                        model=self.model_name
                    )
                    output = response.choices[0].message.content
                    reasoning = response.choices[0].message.reasoning
                    print(reasoning)
                elif 'gemini' in self.model_name:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        config=types.GenerateContentConfig(
                            system_instruction=self.system_prompt
                        ),
                        contents=prompt
                    )
                    output = response.text
                    print('==gemini==')
                    print(output)
                elif 'gpt' in self.model_name:
                    response = self.client.responses.create(model=self.model_name, input=self.system_prompt+'\n\n'+prompt)
                    output = response.output_text
                    print('====gpt=====')
                    print(output)
                else:
                    output = ''

            else:
                output = None
                if 'oss' in self.model_name:
                    convo = Conversation.from_messages(
                        [
                            Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
                            Message.from_role_and_content(
                                Role.DEVELOPER,
                                DeveloperContent.new().with_instructions(self.system_prompt),
                            ),
                            Message.from_role_and_content(Role.USER, prompt),
                        ]
                    )

                    prefill_ids = self.encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
                    outputs = self.model.generate(
                        prompts=[{"prompt_token_ids": prefill_ids}],  # batch of size 1
                        sampling_params=self.sampling,
                    )

                    gen = outputs[0].outputs[0]
                    output_tokens = gen.token_ids  # <-- these are the completion token IDs (no prefill)
                    entries = self.encoding.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)
                    for message in entries:

                        d = message.to_dict()

                        print(json.dumps(d))

                        if d.get("channel") == "final":
                            output = d["content"][0]["text"].strip().upper()
                elif ('qwen' in self.model_name.lower() and self.transformer==False) or 'mistral' in self.model_name.lower() or 'llama' in self.model_name.lower():
                    messages = [
                        {
                            "role": "system",
                            "content": self.system_prompt,
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ]

                    outputs = self.model.chat(
                        messages,
                        sampling_params=self.sampling,
                        chat_template_kwargs={"enable_thinking": True,
                                              "reasoning_effort": "high"}
                    )

                    print(outputs)
                    generated_text = outputs[0].outputs[0].text
                    print('=====llama=====')
                    print(generated_text)
                    if '</think>' in generated_text:
                        print(generated_text)
                        generated_text = generated_text.split('</think>')[-1]

                    output = generated_text.strip().upper()
                    print('===output===')
                    print(output)
                elif 'qwen' in self.model_name.lower() and self.transformer == True:
                    messages = [
                        {"role": "user", "content": self.system_prompt+prompt}
                    ]
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=True  # True is the default value for enable_thinking
                    )
                    model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

                    generated_ids = self.model.generate(
                        **model_inputs,
                        max_new_tokens=10000,
                        temperature=0.6,
                        top_p=0.95,
                        top_k=20,
                        do_sample=True
                    )
                    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

                    # parsing thinking content
                    try:

                        index = len(output_ids) - output_ids[::-1].index(151668)
                    except ValueError:
                        index = 0

                    thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
                    output = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
                    print(thinking_content)
                else:
                    outputs = self.model(
                        messages,
                        max_new_tokens=1024,
                    )
                    output = outputs[0]["generated_text"][-1]['content'].strip().upper()

                print('model')
                print(output)
            if output is None:
                continue

            if 'PLAY' in output or 'WAIT' in output:
                break
        else:
            return 'WAIT'

        if 'WAIT' in output:
            return 'WAIT'
        return 'PLAY'


class RandomPlayer:
    def __init__(self, agent_id, time_unit='second', calibration=False):
        self.agent_id = agent_id
        self.alpha = 0.5
        self.observed_alpha = 0.5
        self.calibration = calibration
        self.time_unit = time_unit

    def estimate_alpha(self, cards_played_info):

        action = cards_played_info['action_num']
        if cards_played_info['action'][action]['topBefore']:
            return cards_played_info['action'][action]['hesitationTime'] / (
                        cards_played_info['action'][action]['value'] - cards_played_info['action'][action]['topBefore'])
        else:
            return cards_played_info['action'][action]['hesitationTime'] / (
            cards_played_info['action'][action]['value'])

    def __str__(self):
        return 'RandomPlayer'

    def decide(self, cards_played_info):
        if cards_played_info['time_passed_last_play']>=1:

            return random.choice(["PLAY", "WAIT"])
        else:
            return 'WAIT'


class RuleBasedAgent:
    def __init__(self, agent_id, alpha, calibration, epsilon=0, lr=0.1):
        self.agent_id = agent_id
        self.original_alpha = alpha
        self.alpha = alpha
        self.observed_alpha = alpha
        self.lr = lr
        self.epsilon = epsilon
        self.calibration = calibration

    def __str__(self):
        return f"RuleBasedAgent{self.original_alpha}_{self.calibration}"

    def estimate_alpha(self, cards_played_info):
        action = cards_played_info['action_num']
        if cards_played_info['action'][action]['topBefore']:
            return cards_played_info['action'][action]['hesitationTime'] / (
                        cards_played_info['action'][action]['value'] - cards_played_info['action'][action]['topBefore'])
        else:
            return cards_played_info['action'][action]['hesitationTime'] / (
            cards_played_info['action'][action]['value'])

    def decide(self, cards_played_info):
        cards_played = cards_played_info['cards_played']
        # print(cards_played_info)
        level = cards_played_info['level']
        partner_id = next(
            pid for pid in cards_played_info['current_hand']
            if pid != self.agent_id
        )
        partner_hand = cards_played_info['current_hand'][partner_id]
        if not partner_hand:
            return 'PLAY'
        current_hand = cards_played_info['current_hand'][self.agent_id]

        if not current_hand:
            return 'WAIT'
        current_played_cards = cards_played[level] if level in cards_played else [0]
        wait_time = (sorted(current_hand)[0] - sorted(current_played_cards)[-1]) * self.alpha

        wait_time_update = cards_played_info['time_passed_last_play']
        if wait_time_update >= wait_time:
            return 'PLAY'
        else:
            return 'WAIT'

    def calibrate(self, cards_played_info):
        actions = sorted(
            [x for x, y in cards_played_info['action'].items() if
             y['level'] == cards_played_info['level'] and y['isError'] != 'AutoPlay'])
        if len(actions) == 0:
            return

        def card(i):
            return cards_played_info['action'][actions[i]]

        cur = card(-1)
        prev = card(-2) if len(actions) >= 2 else None
        level = cards_played_info['level']
        card_played = cards_played_info['cards_played'][level]

        partner_id = next(
            pid for pid in cards_played_info['current_hand']
            if pid != self.agent_id
        )
        partner_hand = cards_played_info['current_hand'][partner_id]
        my_hand = cards_played_info['current_hand'][self.agent_id]

        if cur['player_id'] == self.agent_id and partner_hand == []:
            return
        if cur['player_id'] != self.agent_id and my_hand == []:
            return

        if len(card_played) == int(level) * 2:
            return

        if prev is not None:
            gap = cur['value'] - prev['value']
            hesitation_time = cur['hesitationTime']
        else:
            gap = cur['value']
            hesitation_time = cur['hesitationTime']

        alpha_obs = hesitation_time / gap
        self.alpha = (1 - self.lr) * self.alpha + self.lr * alpha_obs


class BayesianPlayer:
    def __init__(self, agent_id, calibration,  lr=0.5, alpha=0.1, theta=0.9, epsilon=0, gamma=1):
        self.agent_id = agent_id
        self.original_alpha = alpha
        self.alpha = alpha
        self.observed_alpha = alpha
        self.theta = theta
        self.gamma = gamma
        self.epsilon = epsilon
        self.calibration = calibration
        self.lr = lr

    def prior(self, level, played_cards, my_hand, partner_hand, k):
        if level == 1:
            return 1 / 99
        partner_number = len(partner_hand)
        if played_cards:
            left_cards = (set(range(1, 101)) - set(played_cards) - set(my_hand))
        else:
            left_cards = (set(range(1, 101)) - set(my_hand))
        possible_combinations = comb(len(left_cards), partner_number)
        possible_combinations_greater_than_k = comb((len(left_cards) - 1 - len([x for x in left_cards if x < k])),
                                                    (partner_number - 1))

        return possible_combinations_greater_than_k / possible_combinations

    def survival(self, k, t, last_played_card):
        gap = k - last_played_card
        if gap <= 0:
            return 0

        return math.exp(
            - (t / (self.alpha * gap)) ** self.gamma
        )

    def confidence(self, t, level, played_cards, my_hand, card_to_play, partner_hand):
        if played_cards:
            last_card = played_cards[-1]
        else:
            last_card = 0
        if played_cards:
            left_cards = (set(range(1, 101)) - set(played_cards) - set(my_hand))
        else:
            left_cards = (set(range(1, 101)) - set(my_hand))
        nom = sum([self.survival(k, t, last_card) * self.prior(level, played_cards, my_hand, partner_hand, k) for k in
                   left_cards if k > card_to_play])
        denom = sum([self.survival(j, t, last_card) * self.prior(level, played_cards, my_hand, partner_hand, j) for j in
                     left_cards])
        # if denom == 0:
        #     print(
        #         "DENOM ZERO",
        #         t,
        #         self.alpha,
        #         len(left_cards)
        #     )
        return nom / denom if denom != 0 else 0

    def estimate_alpha(self, cards_played_info):
        action = cards_played_info['action_num']
        if cards_played_info['action'][action]['topBefore']:
            return cards_played_info['action'][action]['hesitationTime'] / (
                        cards_played_info['action'][action]['value'] - cards_played_info['action'][action]['topBefore'])
        else:
            return cards_played_info['action'][action]['hesitationTime'] / (
            cards_played_info['action'][action]['value'])

    def __str__(self):
        return f"BayesianAgent{self.original_alpha}_{self.theta}_{self.calibration}"

    def decide(self, cards_played_info):
        t = cards_played_info['time_passed_last_play']
        level = cards_played_info['level']

        current_hand = sorted(cards_played_info['current_hand'][self.agent_id])
        last_cards = sorted(cards_played_info['cards_played'][level]) if level in cards_played_info[
            'cards_played'] else None
        my_hand = sorted(cards_played_info['current_hand'][self.agent_id])
        partner_id = next(
            pid for pid in cards_played_info['current_hand']
            if pid != self.agent_id
        )
        partner_hand = cards_played_info['current_hand'][partner_id]
        if t > 60:
            return 'PLAY'
        if not partner_hand:
            return 'PLAY'
        if not current_hand:
            return 'WAIT'
        else:
            if last_cards:
                last_card = last_cards[-1]
                if current_hand[0] < last_card:
                    return 'PLAY'

            card_to_play = current_hand[0]
            conf = self.confidence(t, level, last_cards, my_hand, card_to_play, partner_hand)
            if conf > self.theta:
                return 'PLAY'
            else:
                return 'WAIT'

    def calibrate(self, cards_played_info):
        actions = sorted(
            [x for x, y in cards_played_info['action'].items() if
             y['level'] == cards_played_info['level'] and y['isError'] != 'AutoPlay'])
        if len(actions) == 0:
            return

        def card(i):
            return cards_played_info['action'][actions[i]]

        cur = card(-1)
        prev = card(-2) if len(actions) >= 2 else None
        level = cards_played_info['level']
        card_played = cards_played_info['cards_played'][level]

        partner_id = next(
            pid for pid in cards_played_info['current_hand']
            if pid != self.agent_id
        )
        partner_hand = cards_played_info['current_hand'][partner_id]
        my_hand = cards_played_info['current_hand'][self.agent_id]

        if cur['player_id'] == self.agent_id and partner_hand == []:
            return
        if cur['player_id'] != self.agent_id and my_hand == []:
            return

        if len(card_played) == int(level) * 2:
            return

        if prev is not None:
            gap = cur['value'] - prev['value']
            hesitation_time = cur['hesitationTime']
        else:
            gap = cur['value']
            hesitation_time = cur['hesitationTime']

        alpha_obs = hesitation_time / gap

        self.alpha = max(0.05, (1 - self.lr) * self.alpha + self.lr * alpha_obs)


class RuleBasedAgentTime:
    def __init__(self, agent_id, alpha, calibration, epsilon=0, lr=0.1):
        self.agent_id = agent_id
        self.original_alpha = alpha
        self.alpha = alpha
        self.observed_alpha = alpha
        self.lr = lr
        self.epsilon = epsilon
        self.calibration = calibration

    def __str__(self):
        if self.lr:
            return f"RuleBasedAgent{self.original_alpha}_{self.calibration}_{self.lr}"
        else:
            return f"RuleBasedAgent{self.original_alpha}_{self.calibration}_0"

    def decide(self, cards_played_info):
        cards_played = cards_played_info['cards_played']
        # print(cards_played_info)
        level = cards_played_info['level']
        partner_id = next(
            pid for pid in cards_played_info['current_hand']
            if pid != self.agent_id
        )
        partner_hand = cards_played_info['current_hand'][partner_id]
        if not partner_hand:
            return 0
        current_hand = cards_played_info['current_hand'][self.agent_id]

        if not current_hand:
            return math.inf
        current_played_cards = cards_played[level] if level in cards_played else [0]
        wait_time = (sorted(current_hand)[0] - sorted(current_played_cards)[-1]) * self.alpha
        return wait_time

    def estimate_alpha(self, cards_played_info):
        action = cards_played_info['action_num']
        if cards_played_info['action'][action]['topBefore']:
            return cards_played_info['action'][action]['hesitationTime'] / (
                        cards_played_info['action'][action]['value'] - cards_played_info['action'][action]['topBefore'])
        else:
            return cards_played_info['action'][action]['hesitationTime'] / (
            cards_played_info['action'][action]['value'])

    def calibrate(self, cards_played_info):
        actions = sorted(
            [x for x, y in cards_played_info['action'].items() if
             y['level'] == cards_played_info['level'] and y['isError'] != 'AutoPlay'])
        if len(actions) == 0:
            return

        def card(i):
            return cards_played_info['action'][actions[i]]

        cur = card(-1)
        prev = card(-2) if len(actions) >= 2 else None
        level = cards_played_info['level']
        card_played = cards_played_info['cards_played'][level]

        partner_id = next(
            pid for pid in cards_played_info['current_hand']
            if pid != self.agent_id
        )
        partner_hand = cards_played_info['current_hand'][partner_id]
        my_hand = cards_played_info['current_hand'][self.agent_id]

        if cur['player_id'] == self.agent_id and partner_hand == []:
            return
        if cur['player_id'] != self.agent_id and my_hand == []:
            return

        if len(card_played) == int(level) * 2:
            return

        if prev is not None:
            gap = cur['value'] - prev['value']
            hesitation_time = cur['hesitationTime']
        else:
            gap = cur['value']
            hesitation_time = cur['hesitationTime']

        alpha_obs = hesitation_time / gap
        # print(hesitation_time, gap)
        self.alpha = (1 - self.lr) * self.alpha + self.lr * alpha_obs


class BayesianPlayerTime:
    def __init__(self, agent_id, calibration, lr=0.5, alpha=0.1, theta=0.9, epsilon=0, gamma=1):
        self.agent_id = agent_id
        self.original_alpha = alpha
        self.alpha = alpha
        self.observed_alpha = alpha
        self.theta = theta
        self.gamma = gamma
        self.epsilon = epsilon
        self.calibration = calibration
        self.lr = lr

    def prior(self, level, played_cards, my_hand, partner_hand, k):
        if level == 1:
            return 1 / 99
        partner_number = len(partner_hand)
        if played_cards:
            left_cards = (set(range(1, 101)) - set(played_cards) - set(my_hand))
        else:
            left_cards = (set(range(1, 101)) - set(my_hand))
        possible_combinations = comb(len(left_cards), partner_number)
        possible_combinations_greater_than_k = comb((len(left_cards) - 1 - len([x for x in left_cards if x < k])),
                                                    (partner_number - 1))

        return possible_combinations_greater_than_k / possible_combinations

    def survival(self, k, t, last_played_card):
        gap = k - last_played_card
        if gap <= 0:
            return 0
        return math.exp(
            - (t / (self.alpha * gap)) ** self.gamma
        )

    def confidence(self, t, level, played_cards, my_hand, card_to_play, partner_hand):
        if played_cards:
            last_card = sorted(played_cards)[-1]
        else:
            last_card = 0
        if played_cards:
            left_cards = (set(range(1, 101)) - set(played_cards) - set(my_hand))
        else:
            left_cards = (set(range(1, 101)) - set(my_hand))
        nom = sum([self.survival(k, t, last_card) * self.prior(level, played_cards, my_hand, partner_hand, k) for k in
                   left_cards if k > card_to_play])
        denom = sum([self.survival(j, t, last_card) * self.prior(level, played_cards, my_hand, partner_hand, j) for j in
                     left_cards])

        return nom / denom if denom != 0 else 0

    def estimate_alpha(self, cards_played_info):
        action = cards_played_info['action_num']
        if cards_played_info['action'][action]['topBefore']:
            return cards_played_info['action'][action]['hesitationTime'] / (
                        cards_played_info['action'][action]['value'] - cards_played_info['action'][action]['topBefore'])
        else:
            return cards_played_info['action'][action]['hesitationTime'] / (
            cards_played_info['action'][action]['value'])

    def __str__(self):
        if self.lr:
            return f"BayesianAgent{self.original_alpha}_{self.theta}_{self.gamma}_{self.calibration}_{self.lr}"
        else:
            return f"BayesianAgent{self.original_alpha}_{self.theta}_{self.gamma}_{self.calibration}_0"

    def wait_seconds(self, level, played_cards, my_hand, card_to_play, partner_hand, T_MAX=60, tol=1e-3, max_iter=40):
        if self.confidence(0, level, played_cards, my_hand, card_to_play, partner_hand) >= self.theta:
            return 0.0
        if self.confidence(T_MAX, level, played_cards, my_hand, card_to_play, partner_hand) < self.theta:
            return T_MAX
        lo, hi = 0.0, T_MAX
        for _ in range(max_iter):
            mid = (lo + hi) / 2
            if self.confidence(mid, level, played_cards, my_hand, card_to_play, partner_hand) >= self.theta:
                hi = mid
            else:
                lo = mid
            if hi - lo < tol:
                break
        return hi

    def decide(self, cards_played_info):
        t = cards_played_info['time_passed_last_play']
        level = cards_played_info['level']

        current_hand = sorted(cards_played_info['current_hand'][self.agent_id])
        last_cards = sorted(cards_played_info['cards_played'][level]) if level in cards_played_info[
            'cards_played'] else None
        my_hand = sorted(cards_played_info['current_hand'][self.agent_id])
        partner_id = next(
            pid for pid in cards_played_info['current_hand']
            if pid != self.agent_id
        )
        partner_hand = cards_played_info['current_hand'][partner_id]

        if not partner_hand:
            return 0
        if not current_hand:
            return math.inf
        else:
            if last_cards:
                last_card = last_cards[-1]
                if current_hand[0] < last_card:
                    return 0

            card_to_play = current_hand[0]

            waiting_time = self.wait_seconds(level, last_cards, my_hand, card_to_play, partner_hand)

            return waiting_time

    def calibrate(self, cards_played_info):
        actions = sorted(
            [x for x, y in cards_played_info['action'].items() if
             y['level'] == cards_played_info['level'] and y['isError'] != 'AutoPlay'])
        if len(actions) == 0:
            return

        def card(i):
            return cards_played_info['action'][actions[i]]

        cur = card(-1)
        prev = card(-2) if len(actions) >= 2 else None
        level = cards_played_info['level']
        card_played = cards_played_info['cards_played'][level]

        partner_id = next(
            pid for pid in cards_played_info['current_hand']
            if pid != self.agent_id
        )
        partner_hand = cards_played_info['current_hand'][partner_id]
        my_hand = cards_played_info['current_hand'][self.agent_id]

        if cur['player_id'] == self.agent_id and partner_hand == []:
            return
        if cur['player_id'] != self.agent_id and my_hand == []:
            return

        if len(card_played) == int(level) * 2:
            return

        if prev is not None:
            gap = cur['value'] - prev['value']
            hesitation_time = cur['hesitationTime']
        else:
            gap = cur['value']
            hesitation_time = cur['hesitationTime']

        alpha_obs = hesitation_time / gap

        self.alpha = max(0.05, (1 - self.lr) * self.alpha + self.lr * alpha_obs)


class RandomPlayerTime:
    def __init__(self, agent_id, calibration=False):
        self.agent_id = agent_id
        self.alpha = 0.5

        self.observed_alpha = 0.5
        self.calibration = calibration

    def estimate_alpha(self, cards_played_info):

        action = cards_played_info['action_num']
        if cards_played_info['action'][action]['topBefore']:
            return cards_played_info['action'][action]['hesitationTime'] / (
                        cards_played_info['action'][action]['value'] - cards_played_info['action'][action]['topBefore'])
        else:
            return cards_played_info['action'][action]['hesitationTime'] / (
            cards_played_info['action'][action]['value'])

    def __str__(self):
        return 'RandomPlayer'

    def decide(self, card_info):
        if random.choice(["PLAY", "WAIT"]) == 'PLAY':
            return 0
        else:
            return 1



class AgentTime:
    def __init__(self, agent_id, model_name, feedback, prompt_type,tokenizer=None, transformer=False, model=None, calibration=False, using_API=True):
        self.agent_id = agent_id
        self.model_name = model_name
        self.feedback = feedback
        self.transformer=transformer
        self.tokenizer = tokenizer
        self.through_API = using_API
        self.calibration = calibration
        self.prompt_type = prompt_type
        self.alpha = None
        self.observed_alpha = self.alpha
        self.system_prompt = (
            """
            You and your partner are playing a cooperative card game. This game is similar to a game called The Mind. However, our rules are slightly different, so please read the instructions very carefully.

            You and your partner play together as one team. 
            At each round (called a level), you and your partner are each dealt a new hand from a fresh deck of cards numbered 1 to 100. Cards from previous levels do not carry over or affect the cards dealt in the new level. You receive one card each in level 1, two cards each in level 2, and so on.

            Your shared goal is to arrange your and your partner's cards in ascending order, from lowest to highest. You cannot see your partner’s cards. You only know how many cards they have left and what cards have been played.

            How it works: 
            1. Each level deals you a new hand. In level N, you and your partner each hold N cards.
            2. You can only play your lowest card at any time. When you believe it is lower than every card your partner still holds, play it. There are no turns — whenever you sense your lowest card should go next, you play it.
            3. If you play a card while your partner still holds a lower one, that is an error — you played too early. When this happens the game pauses automatically, and every card lower than the one you just played is revealed and removed from your partner's hand. 
            4. Clear all cards to finish the level and move on. There are 10 levels in total.

            The performance of the game is equally based on **how fast you finish** and **how few errors you make**. 
            In other words, you should play QUICKLY and make as FEW errors as possible.
            \n
            """
        )

        self.last_prompt = ''
        if self.through_API:
            if 'zai' in model_name:
                self.client = Cerebras(api_key=ZAI_API)
            elif 'gemini' in model_name:
                self.client = genai.Client(api_key=GEMINI_API,
                                           http_options={'timeout': 300000})
            elif 'gpt' in model_name.lower():
                self.client = OpenAI(api_key=OPENAI_API_KEY)
        else:
            if 'oss' in model_name:
                self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
                stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()
                self.sampling = SamplingParams(
                    max_tokens=10000,
                    temperature=0.7,
                    stop_token_ids=stop_token_ids,
                )
            elif 'qwen' in model_name.lower() and transformer == False:
                self.sampling = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=10000)
            elif 'mistral' in model_name.lower():
                self.sampling = SamplingParams(temperature=0.7, max_tokens=10000)
            elif 'llama' in model_name.lower():
                self.sampling = SamplingParams(temperature=0.6, max_tokens=10000)
            self.model = model

    def __str__(self):
        return self.model_name

    def prompt_construct(self, cards_played_info):
        prompt = generate_prompt(cards_played_info, self.agent_id, self.feedback, self.prompt_type)
        self.last_prompt = prompt
        return prompt

    def get_prompt(self):
        return self.last_prompt

    def estimate_alpha(self, cards_played_info):
        action = cards_played_info['action_num']
        if cards_played_info['action'][action]['topBefore']:
            return cards_played_info['action'][action]['hesitationTime'] / (
                    cards_played_info['action'][action]['value'] - cards_played_info['action'][action]['topBefore'])
        else:
            return cards_played_info['action'][action]['hesitationTime'] / (
                cards_played_info['action'][action]['value'])

    def decide(self, cards_played_info):
        prompt = self.prompt_construct(cards_played_info)
        max_retries = 5
        messages = [
            {"role": "system", "content": f"{self.system_prompt}"},
            {"role": "user", "content": f"{prompt}"},
        ]

        partner_id = next(
            pid for pid in cards_played_info['current_hand']
            if pid != self.agent_id
        )

        partner_hand = cards_played_info['current_hand'][partner_id]
        if not partner_hand:
            return 0
        current_hand = cards_played_info['current_hand'][self.agent_id]
        if not current_hand:
            return 100

        for _ in range(max_retries):
            if self.through_API:
                if 'zai' in self.model_name:

                    response = self.client.chat.completions.create(
                        messages=messages,
                        model=self.model_name
                    )
                    output = response.choices[0].message.content
                    reasoning = response.choices[0].message.reasoning
                    print(reasoning)
                elif 'gemini' in self.model_name:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        config=types.GenerateContentConfig(
                            system_instruction=self.system_prompt
                        ),
                        contents=prompt
                    )
                    output = response.text
                    print('==gemini==')
                    print(output)
                elif 'gpt' in self.model_name:
                    response = self.client.responses.create(model=self.model_name, input=self.system_prompt+'\n\n'+prompt)
                    output = response.output_text
                    print('====gpt=====')
                    print(output)

                else:
                    output = ''

            else:
                output = None
                if 'oss' in self.model_name:
                    convo = Conversation.from_messages(
                        [
                            Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
                            Message.from_role_and_content(
                                Role.DEVELOPER,
                                DeveloperContent.new().with_instructions(self.system_prompt),
                            ),
                            Message.from_role_and_content(Role.USER, prompt),
                        ]
                    )

                    prefill_ids = self.encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
                    outputs = self.model.generate(
                        prompts=[{"prompt_token_ids": prefill_ids}],  # batch of size 1
                        sampling_params=self.sampling,
                    )

                    gen = outputs[0].outputs[0]
                    output_tokens = gen.token_ids  # <-- these are the completion token IDs (no prefill)
                    entries = self.encoding.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)
                    for message in entries:

                        d = message.to_dict()

                        print(json.dumps(d))

                        if d.get("channel") == "final":
                            output = d["content"][0]["text"].strip().upper()
                elif (
                        'qwen' in self.model_name.lower() and self.transformer == False) or 'mistral' in self.model_name.lower() or 'llama' in self.model_name.lower():
                    messages = [
                        {
                            "role": "system",
                            "content": self.system_prompt,
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ]

                    outputs = self.model.chat(
                        messages,
                        sampling_params=self.sampling,
                        chat_template_kwargs={"enable_thinking": True,
                                              "reasoning_effort": "high"}
                    )

                    print(outputs)
                    generated_text = outputs[0].outputs[0].text
                    print('=====llama=====')
                    print(generated_text)
                    if '</think>' in generated_text:
                        print(generated_text)
                        generated_text = generated_text.split('</think>')[-1]

                    output = generated_text.strip().upper()
                    print('===output===')
                    print(output)
                elif 'qwen' in self.model_name.lower() and self.transformer == True:
                    messages = [
                        {"role": "user", "content": self.system_prompt + prompt}
                    ]
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=True  # True is the default value for enable_thinking
                    )
                    model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

                    generated_ids = self.model.generate(
                        **model_inputs,
                        max_new_tokens=10000,
                        temperature=0.6,
                        top_p=0.95,
                        top_k=20,
                        do_sample=True
                    )
                    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

                    # parsing thinking content
                    try:

                        index = len(output_ids) - output_ids[::-1].index(151668)
                    except ValueError:
                        index = 0

                    thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
                    output = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
                    print(thinking_content)
                else:
                    outputs = self.model(
                        messages,
                        max_new_tokens=1024,
                    )
                    output = outputs[0]["generated_text"][-1]['content'].strip().upper()

                print('model')
                print(output)
            if output is None:
                continue
            if is_number(output):
                break
        else:
            return float(1)

        return float(output)

