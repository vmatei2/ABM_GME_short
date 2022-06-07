import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns


def get_price_history(ticker, start_date, end_date):
    """
    Function to connect to YFinance api and return df of price data for desired time period
    :param ticker:
    :return:
    """
    # getting the historical data
    price_history = ticker.history(start=start_date, end=end_date)
    return price_history


def plot_closing_price_and_volume(data_frame):
    plt.figure(figsize=(12, 10))
    closing_plot = data_frame.Close.plot(legend=True)
    volume_plot = data_frame.Volume.plot(kind='area', secondary_y=True,
                                         alpha=0.6, legend=True)
    closing_plot.margins(0, 0)
    closing_plot.yaxis.set_tick_params(labelsize=18)
    closing_plot.xaxis.set_tick_params(labelsize=18)
    volume_plot.yaxis.set_tick_params(labelsize=18)
    closing_plot.set_ylabel("Closing price", fontsize=20)
    closing_plot.set_xlabel("Date", fontsize=20)
    volume_plot.set_ylabel("Volume (Shares traded)", fontsize=20)
    volume_plot.margins(0, 0)
    plt.grid(True)
    plt.title("GameStop price and volume during short squeeze event", fontsize=20)
    # For some reason the file still thinks it's in ABM_GME_Short.wiki so need to move out to save in images
    plt.savefig("../images/price_volume_gme")
    plt.show()


def barplot_percentages_on_top(df, title, column, xlabel):
    fig, ax = plt.subplots(figsize=(10, 10))
    total_rows = int(len(df))
    sns.countplot(x=column, data=df)
    plt.title(title, fontsize=18)
    plt.ylabel("Count", fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    for p in ax.patches:
        percentage = '{:.2f}'.format(100 * p.get_height() / total_rows)
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(percentage, (x, y), ha='center', va='center', fontsize=11, xytext=(0, 5),
                    textcoords='offset points')
    plt.show()


def line_plot(xvalues, yvalues, title, xlabel, ylabel):
    plt.figure(figsize=(14, 14))
    plt.plot(xvalues, yvalues)
    ax = plt.gca()
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.ylim(0, 1000)

    plt.xticks(rotation=40, fontsize=10)
    plt.yticks(fontsize=12)
    plt.grid()

    temp = ax.xaxis.get_ticklabels()

    # deciding the frequency with which we show the labels on the plot
    every_nth_showed = 400
    temp = list(set(temp) - set(temp[::every_nth_showed]))
    for label in temp:
        label.set_visible(False)
    i = 0

    for x, y in zip(xvalues, yvalues):
        # annotating the plot on every 3rd entry - spaces out a bit more
        if i % every_nth_showed == 0 or i in [1, 4, 7, 11, 17]:
            if i == 0:
                # the user with the most posts skews the plot too much and power-law distribution can not be observed,
                # hence we limit 0 - 1000 and show the first value outside the plot
                y_position = 999
            else:
                y_position = y
            label = y
            plt.annotate(label, (x, y_position), textcoords="offset points",
                         xytext=(0, 10),
                         ha='center', fontsize=12)
        i += 1

    plt.show()


def plot_all_commitments(all_commitments_each_round):
    plt.figure(figsize=(10, 10))
    x = []
    for i in range(len(all_commitments_each_round)):
        x.append(i)
    plt.plot(x, all_commitments_each_round)
    plt.xlabel("Trading Day")
    plt.ylabel("Commitment Values")
    plt.title("Evolution of all agent commitments")
    plt.show()


def plot_average_commitment(commitment_history):
    plt.figure(figsize=(10, 10))
    plt.plot(commitment_history)
    plt.xlabel("Trading Day")
    plt.ylabel("Average Commitment")
    plt.title("Average Commitment History Evolution")
    plt.show()


def plot_commitment_into_groups(commitment_this_round):
    columns = ["Number of agents in group", "Trading day", "Commitment group"]
    commitment_df = pd.DataFrame(commitment_this_round, columns=columns)
    plt.figure(figsize=(10, 10))
    sns.barplot(x="Trading day", y="Number of agents in group", hue="Commitment group", data=commitment_df)
    plt.show()
    stop = 0


if __name__ == '__main__':
    sns.set_style("darkgrid")
    gme_ticker = "GME"
    gme = yf.Ticker(gme_ticker)
    gme_price_history = get_price_history(gme, "2020-12-01", "2021-02-24")
    plot_closing_price_and_volume(gme_price_history)
    gme_institutional_holders = gme.institutional_holders
    print(gme_institutional_holders)
