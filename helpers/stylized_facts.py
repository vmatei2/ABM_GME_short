import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf

def extract_weekend_data_effect(simulation_history):
    returns_dictionary = {}
    price_history = list(simulation_history.values())
    price_history = np.array(price_history)
    returns_history = pct_change(price_history)
    monday_returns = []
    friday_returns = []
    i = 0
    for date, price in simulation_history.items():
        returns_dictionary[date.date()] = returns_history[i]
        i += 1
    plt.figure(figsize=(12, 8))
    dates, returns = zip(*sorted(returns_dictionary.items()))
    returns = np.array(returns)
    dates = list(dates)
    sns.barplot(dates, returns)
    ax = plt.gca()
    temp = ax.xaxis.get_ticklabels()
    temp = list(set(temp) - set(temp[::5]))  # every 5th entry for x ticks
    for label in temp:
        label.set_visible(False)
    for p in ax.patches:
        x = p.get_x() + p.get_width() / 2
        if x % 5 == 0:
            friday_index = x + 4  # x % 5
            # == 0 ==> we have a monday + 4 to keep Friday annotations
            y = round(p.get_height(), 4)
            monday_returns.append(y)
            y_offset = 12
            if y < 0:
                y_offset = -y_offset
            ax.annotate(y, (x, y), ha='center', va='center', fontsize=7, xytext=(0, y_offset)
                        , textcoords="offset points")
        if x == friday_index:
            y = round(p.get_height(), 4)
            friday_returns.append(y)
            y_offset = 3
            if y < 0:
                y_offset = -y_offset
            ax.annotate(y, (x, y), ha='center', va='center', fontsize=7, xytext=(0, y_offset)
                        , textcoords="offset points")
    plt.xticks(rotation=45)
    plt.xlabel("Dates", fontsize=14)
    plt.ylabel("Return Values", fontsize=14)
    plt.title("Returns Across Simulation Days", fontsize=18)
    print("Mean Returns of Monday: ", np.mean(monday_returns))
    print("Mean Returns of Friday: ", np.mean(friday_returns))
    return returns_dictionary




def pct_change(nparray):
    pct = np.zeros_like(nparray)
    pct[1:] = np.diff(nparray) / np.abs(nparray[:-1])
    return pct


def observe_fat_tails_returns_distribution(price_history):
    # arithmetic returns
    log_returns_stack = np.diff(np.log(price_history))
    plt.figure(figsize=(10, 10))
    sns.distplot(log_returns_stack)
    returns_mean = np.mean(log_returns_stack)
    returns_std_dev = np.std(log_returns_stack)
    gaussian_dist = np.random.normal(returns_mean, returns_std_dev, 50)
    sns.distplot(gaussian_dist)
    plt.legend(["Returns Distribution", "Gaussian Distribution"], fontsize=20)
    plt.show()



def plot_simulation_against_real_values(simulation_values, real_values):
    plt.figure(figsize=(10, 10))
    plt.xlabel("Trading Day", fontsize=20)
    plt.ylabel("Price", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.plot(simulation_values, 'r')
    plt.plot(real_values, 'g')
    plt.title("Simulation price vs Observed price", fontsize=22)
    plt.legend(["Simulated Price", "Real Price"])
    plt.show()


def observe_volatility_clustering(price_history):
    log_returns = np.diff(np.log(price_history))
    price_history = np.array(price_history)
    diff = price_history[1:] - price_history[:-1]
    pct_changes = diff / price_history[1:] * 100
    plt.figure(figsize=(10, 10))
    plt.plot(pct_changes)
    plt.xlabel("Trading Day", fontsize=20)
    plt.ylabel("Log returns (%)", fontsize=20)
    plt.title("Log returns plot", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()


def observe_autocorrelation_abs_returns(price_history):
    returns = np.diff(np.log(price_history))
    returns = returns ** 2
    plt.figure(figsize=(10, 10))
    plot_acf(returns, lags=50)
    plt.title("Autocorrelation of returns", fontsize=20)
    plt.xlabel("Lag", fontsize=18)
    plt.show()