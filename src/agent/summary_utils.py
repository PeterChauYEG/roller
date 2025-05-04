import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def log_rolls(rolls):
    print("Rolls: ", rolls)

def log_win_losses(w, l):
    print("\n======== Summary of wins - losses ========")
    print("Wins: ", w)
    print("Losses: ", l)
    if w + l != 0:
        print("Winrate: ", round(w / (w + l) * 100, 2), "%")
    else:
        print("No games completed")

def log_stats(label, data):
    data = np.array(data)

    best = data.max()
    worst = data.min()
    median = np.median(data)
    mean = data.mean()
    mean = round(mean, 2)

    std = np.std(data)
    std = round(std, 2)

    print("\n======== Summary of {0} ========".format(label))
    print("Best: ", best)
    print("Worst: ", worst)
    print("Mean: ", mean)
    print("Median: ", median)
    print("Std: ", std)

    return mean, median, best, worst

def plot_histogram(title, xlabel, ylabel, mean, median, best, worst, data):
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
    plt.show()

def log_summary(w, l, diffs, damage_dealt, damage_taken, rolls, hands):
    log_win_losses(w, l)
    log_rolls(rolls)

    diff_mean, diff_median, diff_best, diff_worst = log_stats("Difference of damage dealt vs taken (%)", diffs)
    damage_dealt_mean, damage_dealt_median, damage_dealt_best, damage_dealt_worst = log_stats("Damage dealt", damage_dealt)
    damage_taken_mean, damage_taken_median, damage_taken_best, damage_taken_worst = log_stats("Damage taken", damage_taken)
    hands_mean, hands_median, hands_best, hands_worst = log_stats("Hands / game", hands)

    plot_histogram("Difference of damage dealt vs taken (%)", "Difference (%)", "Frequency", diff_mean, diff_median, diff_best, diff_worst, diffs)
    plot_histogram("Damage dealt", "Damage dealt", "Frequency", damage_dealt_mean, damage_dealt_median, damage_dealt_best, damage_dealt_worst, damage_dealt)
    plot_histogram("Damage taken", "Damage taken", "Frequency", damage_taken_mean, damage_taken_median, damage_taken_best, damage_taken_worst, damage_taken)
    plot_histogram("Hands / game", "Hands", "Frequency", hands_mean, hands_median, hands_best, hands_worst, hands)
