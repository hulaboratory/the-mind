import os
from itertools import product
from tqdm import tqdm

from game_realtime import TheMindGame
from agent import (
    BayesianPlayerTime,
    RuleBasedAgentTime,
    RandomPlayerTime
)

RESULT_PATH = "simulation"
os.makedirs(RESULT_PATH, exist_ok=True)

alpha_values = [
    0.1, 0.3, 0.5, 0.7,
    1.0, 1.3, 1.5
]

gamma_values = [
    0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 3.0
]

theta_values = [
    0.5,0.6, 0.7, 0.8, 0.9
]

lr_values = [
    0.1, 0.3, 0.5, 0.7, 0.9
]


player1_types = [
    {
        "name": "random",
        "ptype": "random",
        "calibration": False
    },

    {
        "name": "countbased",
        "ptype": "countbased",
        "calibration": False
    },

    {
        "name": "countbased",
        "ptype": "countbased",
        "calibration": True
    },

    {
        "name": "bayesian",
        "ptype": "bayesian",
        "calibration": False
    },

    {
        "name": "bayesian",
        "ptype": "bayesian",
        "calibration": True
    }
]


player2_configs = [

    {
        "name": "random",
        "ptype": "random",
        "calibration": False,

        "alpha": None,
        "gamma": None,
        "theta": None,
        "lr": None
    },

    {
        "name": "countbased",
        "ptype": "countbased",
        "calibration": False,

        "alpha": 0.5,
        "gamma": None,
        "theta": None,
        "lr": 0
    },

    {
        "name": "countbased",
        "ptype": "countbased",
        "calibration": True,

        "alpha": 0.5,
        "gamma": None,
        "theta": None,
        "lr": 0.3
    },

    {
        "name": "bayesian",
        "ptype": "bayesian",
        "calibration": False,

        "alpha": 0.5,
        "gamma": 2.0,
        "theta": 0.85,
        "lr": 0
    },

    {
        "name": "bayesian",
        "ptype": "bayesian",
        "calibration": True,

        "alpha": 0.5,
        "gamma": 2.0,
        "theta": 0.85,
        "lr": 0.3
    }
]

def make_player(
    name,
    ptype,
    calibration,
    alpha=None,
    gamma=None,
    theta=None,
    lr=None
):

    if ptype == "bayesian":

        return BayesianPlayerTime(
            name,
            alpha=alpha,
            lr=lr,
            gamma=gamma,
            theta=theta,
            calibration=calibration
        )

    elif ptype == "countbased":

        return RuleBasedAgentTime(
            name,
            alpha=alpha,
            lr=lr,
            calibration=calibration
        )

    elif ptype == "random":

        return RandomPlayerTime(name)

    else:

        raise ValueError(
            f"Unknown player type: {ptype}"
        )


player1_configs = []


for model in player1_types:

    name = model["name"]
    ptype = model["ptype"]
    calibration = model["calibration"]

    if ptype == "random":

        player1_configs.append({
            "name": name,
            "ptype": ptype,
            "calibration": calibration,

            "alpha": None,
            "gamma": None,
            "theta": None,
            "lr": None
        })

    elif ptype == "countbased":
        if calibration:

            for alpha, lr in product(alpha_values, lr_values):

                player1_configs.append({
                    "name": name,
                    "ptype": ptype,
                    "calibration": calibration,

                    "alpha": alpha,
                    "gamma": None,
                    "theta": None,
                    "lr": lr
                })
        else:
            for alpha in alpha_values:
                player1_configs.append({
                    "name": name,
                    "ptype": ptype,
                    "calibration": calibration,

                    "alpha": alpha,
                    "gamma": None,
                    "theta": None,
                    "lr": None
                })
    elif ptype == "bayesian":
        if calibration:
            for alpha, gamma, theta, lr in product(
                alpha_values,
                gamma_values,
                theta_values,
                lr_values
            ):

                player1_configs.append({
                    "name": name,
                    "ptype": ptype,
                    "calibration": calibration,

                    "alpha": alpha,
                    "gamma": gamma,
                    "theta": theta,
                    "lr": lr
                })
        else:
            for alpha, gamma, theta in product(
                alpha_values,
                gamma_values,
                theta_values
            ):

                player1_configs.append({
                    "name": name,
                    "ptype": ptype,
                    "calibration": calibration,

                    "alpha": alpha,
                    "gamma": gamma,
                    "theta": theta,
                    "lr": None
                })


print(
    f"Player 1 configurations: "
    f"{len(player1_configs)}"
)

print(
    f"Player 2 configurations: "
    f"{len(player2_configs)}"
)

print(
    f"Total configuration pairs: "
    f"{len(player1_configs) * len(player2_configs)}"
)

print(
    f"Total games: "
    f"{len(player1_configs) * len(player2_configs) * 10}"
)


# ============================================================
# Run simulations
# ============================================================

for p1 in tqdm(
    player1_configs,
    desc="Player 1 configurations"
):

    for p2 in player2_configs:

        # ----------------------------------------------------
        # Create readable configuration name
        # ----------------------------------------------------

        p1_name = p1["name"]

        if p1["ptype"] == "random":

            p1_label = "random"

        elif p1["ptype"] == "countbased":

            p1_label = (
                f"countbased_"
                f"cal{p1['calibration']}_"
                f"a{p1['alpha']}"
            )

        else:

            p1_label = (
                f"bayesian_"
                f"cal{p1['calibration']}_"
                f"a{p1['alpha']}_"
                f"g{p1['gamma']}_"
                f"t{p1['theta']}_"
                f"lr{p1['lr']}"
            )


        if p2["ptype"] == "random":

            p2_label = "random"

        elif p2["ptype"] == "countbased":

            p2_label = (
                f"countbased_"
                f"cal{p2['calibration']}_"
                f"a{p2['alpha']}"
            )

        else:

            p2_label = (
                f"bayesian_"
                f"cal{p2['calibration']}_"
                f"a{p2['alpha']}_"
                f"g{p2['gamma']}_"
                f"t{p2['theta']}_"
                f"lr{p2['lr']}"
            )

        for hand in range(1, 11):

            player1 = make_player(
                "player1",
                p1["ptype"],
                p1["calibration"],
                p1["alpha"],
                p1["gamma"],
                p1["theta"],
                p1["lr"]
            )


            player2 = make_player(
                "player2",
                p2["ptype"],
                p2["calibration"],
                p2["alpha"],
                p2["gamma"],
                p2["theta"],
                p2["lr"]
            )


            players = {
                "player1": player1,
                "player2": player2
            }

            game = TheMindGame(
                players,
                RESULT_PATH,
                hand=hand
            )

            game.game_play()