import numpy as np


def bot_param(difficulty):
    if difficulty >= 100:
        return 100
    param = int(np.random.normal(difficulty, 10))
    param = min(param, 100)
    param = max(param, 1)
    return param


def fmt_bot(difficulty, idx):
    weaponpref = list(range(9))
    np.random.shuffle(weaponpref)
    # weaponpref_str = "".join([str(w) for w in weaponpref])  # currently not used

    bot_str = (
        "{ \n"
        f"    name BOT_{difficulty}_{idx} \n"
        f"    aiming {bot_param(difficulty)} \n"
        f"    perfection {bot_param(difficulty)} \n"
        f"    reaction {bot_param(difficulty)} \n"
        f"    isp {bot_param(difficulty)} \n"
        f'    color "00 ff 00" \n'
        f"    skin base \n"
        f"    //weaponpref	012345678 \n"
        "} \n"
        "\n"
    )
    return bot_str


def main():
    levels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for level in levels:
        for idx in range(10):
            bot_cfg_string = fmt_bot(level, idx)
            print(bot_cfg_string)


if __name__ == "__main__":
    main()
