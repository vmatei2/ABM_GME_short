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

if __name__ == '__main__':
    subscribers_numbers = [1580000, 1700000, 2060000, 8060000, 9620000, 10500000]
    growth_percentages = calculate_subscriber_monthly_growth(subscribers_numbers)
    print(growth_percentages)