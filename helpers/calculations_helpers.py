from datetime import datetime

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


if __name__ == '__main__':
    print_current_time()
