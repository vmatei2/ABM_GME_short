import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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
    log_returns = np.log(np.diff(price_history))
    plt.figure(figsize=(10, 10))
    ii = np.isfinite(log_returns) # getting rid of NaN values
    log_returns = log_returns[ii]
    sns.distplot(log_returns)
    returns_mean = np.mean(log_returns)
    returns_std_dev = np.std(log_returns)
    gaussian_dist = np.random.normal(returns_mean, returns_std_dev, 50)
    sns.distplot(gaussian_dist)
    plt.legend(["Returns Distribution", "Gaussian Distribution"], fontsize=20)
    plt.show()



def calculate_log_returns(price_history):
    price_history = np.array(price_history)
    percentage_change = pct_change(price_history)
    log_returns = np.log(1 + percentage_change)  # log returns are simply the natural log of 1 plus the arithmetic
    return log_returns


def plot_simulation_against_real_values(simulation_values, real_values):
    plt.figure(figsize=(10, 10))
    plt.xlabel("Date", fontsize=20)
    plt.ylabel("Price", fontsize=20)
    plt.plot(simulation_values, 'r')
    plt.plot(real_values, 'g')
    plt.title("Simulation price vs Observed price", fontsize=22)
    plt.legend(["Simulated Price", "Real Price"])
    plt.show()
