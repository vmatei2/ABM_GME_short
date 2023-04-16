import math
import random
from datetime import datetime
import seaborn as sns
sns.set_style("darkgrid")
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np


def print_current_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time = ", current_time)


def calculate_subscriber_monthly_growth(numbers):
    growth_percentages = []
    for i in range(len(numbers)):
        if i != len(numbers) - 1:
            this_month = numbers[i]
            next_month = numbers[i + 1]
            difference = next_month - this_month
            percentage_growth = difference * 100 / this_month
            growth_percentages.append(percentage_growth)
    return growth_percentages


def split_commitment_into_groups(commitment_this_round, trading_day):
    zero_to_40 = [x for x in commitment_this_round if x < 0.4]
    forty_to_65 = [x for x in commitment_this_round if 0.4 <= x < 0.65]
    sixty_five_to_one = [x for x in commitment_this_round if 0.65 <= x <= 1]
    zero_to_40_list = [len(zero_to_40), trading_day, "0-0.4"]
    forty_to_65_list = [len(forty_to_65), trading_day, "0.4-0.65"]
    sixty_five_to_one_list = [len(sixty_five_to_one), trading_day, "0.65-1"]
    return zero_to_40_list, forty_to_65_list, sixty_five_to_one_list


def extract_values_counts_as_lists(df, column, sort=True):
    values = df[column].value_counts(dropna=False, sort=sort).keys().tolist()
    counts = df[column].value_counts(dropna=False, sort=sort).tolist()
    return values, counts


def rescale_array(original_array):
    max = np.max(original_array)
    min = np.min(original_array)
    scaled_array = np.array([(x - min) / (max - min) for x in original_array])
    return scaled_array


def check_and_convert_imaginary_number(number):
    if isinstance(number, complex):
        return number.real
    return number


def calculate_d1(S, K, r, volatility, T, t=0):
    d1 = (np.log(S / K) + (r + (volatility ** 2) / 2) * T) / (T - t)
    return d1


def calculate_pdf(d1):
    pdf = (math.e ** (-math.pow(d1, 2) / 2)) / (np.sqrt(2 * np.pi))
    return pdf


def calculate_gamma(S, K, r, volatility, T):
    d1 = calculate_d1(S, K, r, volatility, T)
    pdf = calculate_pdf(d1)
    gamma = pdf / (S * volatility * np.sqrt(T))
    return gamma


def gamma_variation(K, r, volatility, T):
    all_gammas = []
    stock_prices = []
    for stock_price in range(0, int(2.5 * K)):
        gamma = calculate_gamma(stock_price, K, r, volatility, T=T)
        all_gammas.append(gamma)
        stock_prices.append(stock_price)
    return all_gammas, stock_prices


def plot_gamma_variation(all_gammas, stock_prices):
    plt.figure(figsize=(10, 10))
    plt.plot(stock_prices, all_gammas)
    plt.xlabel("Stock Price", fontsize=18)
    plt.ylabel("Gamma ($\Gamma$)", fontsize=18)
    plt.xticks([i for i in stock_prices if i % 10 == 0], fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim(0, 140)
    plt.title("Variation of $\Gamma$ with stock price for an option with $K=60$", fontsize=20)
    plt.savefig("../images/variation_of_gamma")
    plt.show()


def probably(chance):
    """
    Function to return True / False based on given proability
    :param chance: has to be in [0,1] interval
    e.g. we want 10% probability --> probably(0.1)
    :return:
    """
    return random.random() < chance


if __name__ == '__main__':
    all_gammas, stock_prices = gamma_variation(K=60, r=0.05, volatility=0.2, T=0.19) # 0.19 = fraction 10 weeks of a year
    plot_gamma_variation(all_gammas, stock_prices)

    test_gamma = calculate_gamma(S=49, K=50, r=0.05, volatility=0.2, T=0.19)
    print(test_gamma)
