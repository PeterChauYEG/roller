import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns


def log_rolls(rolls: int) -> None:
    print("Rolls: ", rolls)


def log_win_losses(wins: int, losses: int) -> None:
    print("\n======== Summary of wins - losses ========")
    print("Wins: ", wins)
    print("Losses: ", losses)
    if wins + losses != 0:
        print("Winrate: ", round(wins / (wins + losses) * 100, 2), "%")
    else:
        print("No games completed")


def log_stats(label, data):
    print("\n======== Summary of {0} ========".format(label))

    if len(data) == 0:
        print(f"No {label}")
        return 0, 0, 0, 0

    data = np.array(data)

    best = data.max()
    worst = data.min()
    median = np.median(data)
    mean = data.mean()
    mean = round(mean, 2)

    std = np.std(data)
    std = round(std, 2)

    print("Best: ", best)
    print("Worst: ", worst)
    print("Mean: ", mean)
    print("Median: ", median)
    print("Std: ", std)

    return mean, median, best, worst


def plot_histogram(title, xlabel, ylabel, mean, median, best, worst, data):
    if len(data) == 0:
        return

    sns.set_style("darkgrid")
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.hist(data, bins=20, color="blue", alpha=0.7)
    plt.axvline(mean, color="red", linestyle="dashed", linewidth=1)
    plt.axvline(median, color="orange", linestyle="dashed", linewidth=1)
    plt.axvline(best, color="green", linestyle="dashed", linewidth=1)
    plt.axvline(worst, color="purple", linestyle="dashed", linewidth=1)
    plt.legend({"Mean": mean, "Median": median, "Best": best, "Worst": worst})
    plt.savefig(f"{title}.png")
    plt.show()


def log_summary(
    wins, losses, diffs, damage_dealt, damage_taken, rolls, hands, battles_won
):
    log_win_losses(wins, losses)
    log_rolls(rolls)

    diff_mean, diff_median, diff_best, diff_worst = log_stats(
        "Difference of damage dealt vs taken (%)", diffs
    )
    (
        damage_dealt_mean,
        damage_dealt_median,
        damage_dealt_best,
        damage_dealt_worst,
    ) = log_stats("Damage dealt", damage_dealt)
    (
        damage_taken_mean,
        damage_taken_median,
        damage_taken_best,
        damage_taken_worst,
    ) = log_stats("Damage taken", damage_taken)
    hands_mean, hands_median, hands_best, hands_worst = log_stats(
        "Hands / game", hands
    )
    (
        battles_won_mean,
        battles_won_median,
        battles_won_best,
        battles_won_worst,
    ) = log_stats("Battles won", battles_won)

    plot_histogram(
        "Difference of damage dealt vs taken",
        "Difference (%)",
        "Frequency",
        diff_mean,
        diff_median,
        diff_best,
        diff_worst,
        diffs,
    )
    plot_histogram(
        "Damage dealt",
        "Damage dealt",
        "Frequency",
        damage_dealt_mean,
        damage_dealt_median,
        damage_dealt_best,
        damage_dealt_worst,
        damage_dealt,
    )
    plot_histogram(
        "Damage taken",
        "Damage taken",
        "Frequency",
        damage_taken_mean,
        damage_taken_median,
        damage_taken_best,
        damage_taken_worst,
        damage_taken,
    )
    plot_histogram(
        "Hands per game",
        "Hands",
        "Frequency",
        hands_mean,
        hands_median,
        hands_best,
        hands_worst,
        hands,
    )
    plot_histogram(
        "Battles won",
        "Battles won",
        "Frequency",
        battles_won_mean,
        battles_won_median,
        battles_won_best,
        battles_won_worst,
        battles_won,
    )
