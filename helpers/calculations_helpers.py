def calculate_subscriber_monthly_growth(numbers):
    growth_percentages = []
    for i in range(len(numbers)):
        if i != len(numbers) - 1:
            this_month = numbers[i]
            next_month = numbers[i+1]
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


if __name__ == '__main__':
    subscribers_numbers = [1580000, 1700000, 2060000, 8060000, 9620000, 10500000]
    growth_percentages = calculate_subscriber_monthly_growth(subscribers_numbers)
    print(growth_percentages)