import base64
from openai import OpenAI
from playwright.sync_api import sync_playwright
from rich.console import Console
from rich.panel import Panel
import argparse
console = Console()

OPENAI_API_KEY=''
the_mind_page = ''
client = OpenAI(api_key=OPENAI_API_KEY)

SCREEN_WIDTH = 1440
SCREEN_HEIGHT = 900
MODEL = "gpt-5.6-luna"
MAX_TURNS = 400

COMPUTER_TOOL = {
    "type": "computer"
}

PAUSE ={
    "type": "function",
    "name": "pause",
    "description": (
        "Wait for a specified duration when you are not ready to play the card."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "seconds": {
                "type": "number",
            }
        },
        "required": ["seconds"],
        "additionalProperties": False
    }
}


USER_PROMPT='''
Finish the game following the instructions and report the results in the end.

'''
SYSTEM_PROMPT = '''
            Your goal is to play the following game in the browser.

            You and your partner are playing a cooperative card game. This game is similar to a game called The Mind. However, our rules are slightly different, so please read the instructions very carefully.
            You and your partner play together as one team.
            At each round (called a level, 10 levels in total), you and your partner are each dealt a new hand from a fresh deck of cards numbered 1 to 100. Cards from previous levels do not carry over or affect the cards dealt in the new level. You receive one card each in level 1, two cards each in level 2, and so on.

            Your shared goal is to arrange your and your partner's cards in ascending order, from lowest to highest. You cannot see your partner’s cards. You only know how many cards they have left and what cards have been played.

            How it works:
            1. Each level deals you a new hand. In level N, you and your partner each hold N cards.
            2. You can only play your lowest card at any time. When you believe it is lower than every card your partner still holds, play it. There are no turns — whenever you sense your lowest card should go next, you play it.
            3. If you play a card while your partner still holds a lower one, that is an error — you played too early. When this happens the game pauses automatically, and every card lower than the one you just played is revealed and removed from your partner's hand.
            4. Clear all cards to finish the level and move on. There are 10 levels in total.

            The performance of the game is equally based on **how fast you finish** and **how few errors you make**.
            In other words, you should play QUICKLY and make as FEW errors as possible. You can decide to wait or click to play. 


            *** WAITING POLICY***: If you choose to wait, one second will pass.
            
            Here is an introduction of the game interface:
            The game interface is divided into four main panels.
            The top-left panel displays the partner’s cards, while the top-right panel shows the player’s own cards and provides the "Play card" or "continue" button for playing a card or continue to the next level or play.
            The bottom-left panel presents game information, including the current level, number of errors, number of cards remaining, and the highest card played, as well as an event log.
            The bottom-right panel displays the cards that have already been played.
            '''


def print_reasoning(response):
    for item in response.output:
        if getattr(item, "type", None) != "reasoning":
            continue

        for summary in getattr(item, "summary", []) or []:
            text = getattr(summary, "text", None)
            if text:
                console.print(
                    Panel(
                        text,
                        title="[bold yellow]Model Reasoning[/]",
                        border_style="yellow",
                        expand=False,
                    )
                )


def screenshot_data_url(page):
    image_bytes = page.screenshot(type="png")
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{image_b64}"

def get_computer_call(response):
    calls = [
        item for item in response.output
        if item.type == "computer_call"
    ]


    return calls[0] if calls else None


def execute_openai_action(page, action):
    action_type = action.type

    if action_type == "keypress":
        key_map = {
            "ENTER": "Enter",
            "RETURN": "Enter",
            "ESC": "Escape",
            "ESCAPE": "Escape",
            "TAB": "Tab",
            "SPACE": "Space",
            "BACKSPACE": "Backspace",
            "DELETE": "Delete",
            "ARROWUP": "ArrowUp",
            "ARROWDOWN": "ArrowDown",
            "ARROWLEFT": "ArrowLeft",
            "ARROWRIGHT": "ArrowRight",
        }

        for key in action.keys:
            mapped_key = key_map.get(key.upper(), key)
            page.keyboard.press(mapped_key)

    elif action_type == "click":
        page.mouse.click(action.x, action.y)

    elif action_type == "scroll":
        page.mouse.wheel(
            action.scroll_x,
            action.scroll_y
        )

    elif "wait" in action_type:
        page.wait_for_timeout(int(1 * 1000))

    elif action_type == "screenshot":
        pass

    else:
        console.print("[bold red] Unhandled action:", action_type)

    # page.wait_for_timeout(300)


def execute_actions(page, computer_call, turn):
    actions = computer_call.actions or []

    real_actions = [
        action for action in actions
    ]

    if len(real_actions) > 1:
        console.print(
            "Model returned multiple real actions:",
            [action.type for action in real_actions]
        )

    for index, action in enumerate(real_actions):
        console.print(f"Executing action {index + 1}: {action.type}")
        execute_openai_action(page, action)
        image_bytes = page.screenshot(type="png")

        with open(
            f"debug_turn_{turn + 1}_action_{index + 1}.png",
            "wb"
        ) as f:
            f.write(image_bytes)

    return page.screenshot(type="png")


def main(condition, hand):
    console.print(
        Panel(
            "[bold]GPT Computer Use — Browser Agent[/]\n"
            f"GAME SETTING CONDITION: {condition}, HAND: {hand}",
            border_style="blue",
        )
    )


    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)

        context = browser.new_context(
            viewport={
                "width": SCREEN_WIDTH,
                "height": SCREEN_HEIGHT
            }
        )

        page = context.new_page()
        page.goto(f'{the_mind_page}l?CONDITION_ID={condition}&HAND_ID={hand}')
        first_image = screenshot_data_url(page)

        try:
            response = client.responses.create(
                model=MODEL,
                tools=[COMPUTER_TOOL,
                       PAUSE],
                reasoning={
                    "summary": "concise"
                },
                instructions=SYSTEM_PROMPT,
                input=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": USER_PROMPT
                        },
                        {
                            "type": "input_image",
                            "image_url": first_image
                        }
                    ]
                }]
            )

            for turn in range(MAX_TURNS):
                console.rule(f"[bold magenta]Iteration {turn + 1}[/]")

                print_reasoning(response)
                computer_call = get_computer_call(response)

                if computer_call is None:
                    console.rule("[bold magenta]Agent finished[/]")
                    console.print(response.output_text)
                    break

                screenshot_bytes = execute_actions(
                    page,
                    computer_call,
                    turn
                )

                screenshot_b64 = base64.b64encode(
                    screenshot_bytes
                ).decode("utf-8")

                response = client.responses.create(
                    model=MODEL,
                    previous_response_id=response.id,
                    tools=[COMPUTER_TOOL],
                    reasoning={
                        "summary": "concise"
                    },
                    input=[{
                        "type": "computer_call_output",
                        "call_id": computer_call.call_id,
                        "output": {
                            "type": "computer_screenshot",
                            "image_url": (
                                    "data:image/png;base64,"
                                    + screenshot_b64
                            )
                        }
                    }]
                )
        finally:
            browser.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--condition')
    parser.add_argument('-i', '--hand')

    args = parser.parse_args()
    condition = args.condition
    hand = args.hand
    main(condition, hand)