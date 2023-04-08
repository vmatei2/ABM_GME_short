import collections
import networkx as nx
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import graph_tool.all as gt
from helpers.calculations_helpers import extract_values_counts_as_lists, rescale_array
from helpers.network_helpers import nx2gt


def get_price_history(ticker, start_date, end_date):
    """
    Function to connect to YFinance api and return df of price data for desired time period
    :param ticker:
    :return:
    """
    # getting the historical data
    price_history = ticker.history(start=start_date, end=end_date)
    return price_history


def select_closing_prices(price_history_df):
    closing_prices = price_history_df["Close"].to_list()
    closing_prices = [x * 4 for x in closing_prices]
    return closing_prices


def plot_two_df_columns_together(data_frame, first_column, second_column, third_column=None, fourth_column=None,
                                 kind=None, rescale=False,
                                 title=""):
    plt.figure(figsize=(10, 10))
    if rescale:
        data_frame[first_column] = rescale_array(data_frame[first_column])
        data_frame[second_column] = rescale_array(data_frame[second_column])
        if third_column is not None:
            data_frame[third_column] = rescale_array(data_frame[third_column])
        if fourth_column is not None:
            data_frame[fourth_column] = rescale_array(data_frame[fourth_column])
    first_plot = data_frame[first_column].plot(legend=True)
    if kind == None:
        second_plot = data_frame[second_column].plot(secondary_y=True, alpha=0.6, legend=True)
    else:
        second_plot = data_frame[second_column].plot(kind='area', secondary_y=True,
                                                     alpha=0.6, legend=True)

    if third_column is not None:
        third_plot = data_frame[third_column].plot(legend=True)
    if fourth_column is not None:
        fourth_plot = data_frame[fourth_column].plot(legend=True)
    first_plot.margins(0, 0)
    first_plot.yaxis.set_tick_params(labelsize=18)
    first_plot.xaxis.set_tick_params(labelsize=18)
    second_plot.yaxis.set_tick_params(labelsize=18)
    first_plot.set_ylabel("Closing price", fontsize=20)
    first_plot.set_xlabel("Date", fontsize=20)
    second_plot.set_ylabel("Volume (Shares traded)", fontsize=20)
    second_plot.margins(0, 0)
    plt.grid(True)
    plt.title(title, fontsize=20)
    # For some reason the file still thinks it's in ABM_GME_Short.wiki so need to move out to save in images
    plt.savefig("../images/" + title)
    plt.show()


def barplot_percentages_on_top(df, title, column, xlabel):
    fig, ax = plt.subplots(figsize=(7, 10))
    total_rows = int(len(df))
    sns.countplot(x=column, data=df)
    plt.title(title, fontsize=20)
    plt.ylabel("Count", fontsize=18)
    plt.xlabel(xlabel, fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    for p in ax.patches:
        percentage = '{:.2f}'.format(100 * p.get_height() / total_rows)
        percentage = percentage + "%"
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(percentage, (x, y), ha='center', va='center', fontsize=18, xytext=(0, 5),
                    textcoords='offset points')
    plt.savefig("post_authors_premium_accs.jpg")
    plt.show()


def line_plot(xvalues, yvalues, title, xlabel, ylabel, every_nth_showed, ylim=None):
    plt.figure(figsize=(14, 14))
    plt.plot(xvalues, yvalues)
    ax = plt.gca()
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    if ylim is not None:
        minimum = ylim[0]
        maximum = ylim[1]
        plt.ylim(minimum, maximum)
    plt.xticks(rotation=40, fontsize=10)
    plt.yticks(fontsize=12)
    plt.grid(True)

    temp = ax.xaxis.get_ticklabels()
    # deciding the frequency with which we show the labels on the plot
    every_nth_showed = every_nth_showed
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


def log_log_plot(author_values, xlabel, ylabel, title):
    print("In the log log function")
    hist = dict(collections.Counter(author_values))
    print("Calculated author_hist vector")
    plt.figure(figsize=(7, 10))
    plt.grid(True)
    plt.loglog(hist.keys(), hist.values(), 'ro-')

    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.title(title, fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig("../images/" + title)
    plt.show()


def plot_all_commitments(all_commitments_each_round, number_of_agents, average_commitment_history, title):
    plt.figure(figsize=(8, 8))
    x = []
    for i in range(len(all_commitments_each_round)):
        x.append(i)
        all_commitments_each_round[i] = all_commitments_each_round[i][:number_of_agents]
    plt.plot(x, all_commitments_each_round)
    plt.plot(average_commitment_history, 'bo', label='Average commitment across the network')
    plt.xlabel("Trading Day", fontsize=16)
    plt.ylabel("Commitment Values", fontsize=16)
    plt.title(title, fontsize=20)
    plt.legend(loc='upper right')
    plt.savefig("../images/" + title)
    plt.show()


def simple_line_plot(values_to_be_plotted, xlabel, ylabel, title):
    plt.figure(figsize=(10, 10))
    plt.plot(values_to_be_plotted)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_commitment_into_groups(commitment_this_round, title):
    columns = ["Number of agents in group", "Trading day", "Commitment group"]
    commitment_df = pd.DataFrame(commitment_this_round, columns=columns)
    plt.figure(figsize=(10, 10))
    ax = sns.barplot(x="Trading day", y="Number of agents in group", hue="Commitment group", data=commitment_df)
    for container in ax.containers:
        ax.bar_label(container, fontsize=15)
    plt.title(title, fontsize=20)
    plt.savefig("../images/" + title)
    plt.show()


def plot_institutional_investors_decisions(decision_dict, dates):
    short_gme_decisions = []
    close_positions_decisions = []
    for day, decisions in decision_dict.items():
        short_gme_decisions.append(decisions.count(True))  # sum returns the number of True elements in a boolean array
        close_positions_decisions.append(decisions.count(False))

    plt.plot(dates, short_gme_decisions, 'r')
    plt.plot(dates, close_positions_decisions, 'y')

    plt.xlabel("Trading Day", fontsize=17)
    plt.ylabel("Count", fontsize=17)
    plt.title("Institutional Investor Decisions at each trading day", fontsize=20)
    plt.legend(['Short GME Stock(Take Gamble)', 'Close Short Position(Accept Sure Loss)'], fontsize=12,
               loc=6)  # 6 = center left location
    plt.savefig("../images/institutional_inv_decisions")



####  NETWORK PLOTTING HELPERS


def visualise_network(G, threshold, title, axs, use_graph_tool=False):
    fig = plt.figure(figsize=(8, 8))
    if use_graph_tool:
        graph = nx2gt(G)
        deg = graph.degree_property_map("total")  # undirected graph so we are only working with total degrees
        deg.a = 4 * (np.sqrt(deg.a) * 0.5 + 0.4)
        pos = gt.sfdp_layout(graph)
        control = graph.new_edge_property("vector<double>")
        for e in graph.edges():
            d = np.sqrt(sum((pos[e.source()].a - pos[e.target()].a) ** 2)) / 5  # for curvy edges
            control[e] = [0.3, d, 0.7, d]
        gt.graph_draw(graph, mplfig=fig, pos=pos, vertex_size=deg, vertex_fill_color=deg, vorder=deg,
                      edge_control_points=control)
        plt.savefig("../images/network week " + str(title))
        return
    d = nx.degree(G)
    degree_values = [v for k, v in d]
    nx.draw_networkx(G, pos=nx.spring_layout(G, k=0.99), nodelist=G.nodes(), node_size=[v * 10 for v in degree_values],
                     with_labels=False,
                     node_color='lightgreen', alpha=0.6, ax=axs)
    title = title + 1
    axs.set_title("Network of users (commitment > " + str(threshold) + ") period " + str(title))
    axs.title.set_size(22)


def scale_and_plot(first_array, second_array, title):
    plt.figure(figsize=(10, 10))
    first_array = rescale_array(first_array)
    second_array = rescale_array(second_array)
    plt.plot(first_array, 'ro-')
    plt.plot(second_array, 'bx-')
    plt.ylabel("Value")
    plt.title(title, fontsize=20)
    plt.show()


def plot_demand_dictionary(demand_dict, trading_period):
    all_retail_demand = demand_dict['retail']
    all_hf_demand = demand_dict['institutional']
    plt.figure(figsize=(10, 10))
    plt.plot(trading_period, all_retail_demand, 'g')
    plt.plot(trading_period, all_hf_demand, 'y')

    plt.xlabel("Trading Day")
    plt.ylabel("Demand Evolution")
    legend_ = ["Demand from retail agents", "Demand from institutional investor agents"]
    plt.legend(legend_)
    plt.show()


def plot_hedge_funds_involvment(hf_involved_dict):
    plt.figure(figsize=(10, 10))
    plt.plot(hf_involved_dict['involved'], 'g')
    plt.plot(hf_involved_dict['closed'], 'r')
    plt.grid()
    plt.xlabel('Trading Day')
    plt.ylabel('Number of hedge funds')
    plt.title('Hedge funds involvement throughout simulation')
    plt.show()

def plot_multiple_figures(gme_price_history):
    fig, ax = plt.subplots(2, 2, figsize=(16, 12))

    axes = ax[0, 0]

    title_fontsize = 18
    xticklabel_fontsize = 14

    ax[0, 0].plot(gme_price_history["Close"], 'r')
    ax[0, 0].set_title("Closing price")
    ax[0, 0].title.set_size(title_fontsize)
    ax[0, 0].set_xticklabels(gme_price_history.index.date, rotation=45, fontsize=xticklabel_fontsize)

    ax[0, 1].plot(gme_price_history["Volume"], 'b')
    ax[0, 1].set_title("Volume Traded")
    ax[0, 1].title.set_size(title_fontsize)
    ax[0, 1].set_xticklabels(gme_price_history.index.date, rotation=45, fontsize=xticklabel_fontsize)

    # let's calculate volatility here for the stock
    gme_price_history['Daily Return'] = gme_price_history['Close'].pct_change(1)
    ax[1, 0].plot(gme_price_history["Daily Return"], 'g')
    ax[1, 0].set_title("Daily Return (%)")
    ax[1, 0].title.set_size(title_fontsize)
    ax[1, 0].set_xticklabels(gme_price_history.index.date, rotation=45, fontsize=xticklabel_fontsize)

    # plot combining 3
    gme_price_history["Close"] = rescale_array(gme_price_history["Close"])
    gme_price_history["Volume"] = rescale_array(gme_price_history["Volume"])
    gme_price_history["Open"] = rescale_array(gme_price_history["Open"])
    gme_price_history["Daily Return"] = rescale_array(gme_price_history["Daily Return"])
    ax[1, 1].plot(gme_price_history["Close"], label="Closing Price")
    ax[1, 1].plot(gme_price_history["Volume"], label="Volume")
    ax[1, 1].plot(gme_price_history["Daily Return"], label="Daily Return")
    ax[1, 1].legend()
    ax[1, 1].set_title("Rescaled closing, volume and return values")
    ax[1, 1].title.set_size(title_fontsize)
    ax[1, 1].set_xticklabels(gme_price_history.index.date, rotation=45, fontsize=xticklabel_fontsize)

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)

    plt.savefig("../images/gme_analysis")
    plt.show()


def barplot_options_bought(dates, options_bought):
    fig, ax = plt.subplots(figsize=(8, 8))
    dates = dates[:43]
    dates = [date.date() for date in dates]
    dates_new = []
    for i, date in enumerate(dates):
        if i % 2 == 0:
            dates_new.append(date)
        else:
            dates_new.append(" ")
    options_bought = options_bought[:43]
    sns.barplot(dates, options_bought, ax=ax)
    plt.xlabel("Dates", fontsize=16)
    plt.xticks(rotation=45, fontsize=13)
    plt.yticks(fontsize=13)
    plt.locator_params(axis='x', nbins=len(dates) / 2)
    plt.ylabel("Options Volume", fontsize=16)
    plt.title("Option Trading Volume in Simulation", fontsize=18)
    plt.show()


if __name__ == '__main__':
    sns.set_style("darkgrid")
    gme_ticker = "GME"
    gme = yf.Ticker(gme_ticker)
    gme_price_history = get_price_history(gme, "2020-12-08", "2021-02-04")
    plot_multiple_figures(gme_price_history)
    gme_institutional_holders = gme.institutional_holders
    gme_dates = gme_price_history.index
    gme_dates_as_string = []
    for date in gme_dates:
        date = date.strftime("%Y-%m-%d")
        gme_dates_as_string.append(date)

    cleaned_posts_path = "../kaggleData/cleaned_posts_data_dt_index"
    wsb_posts_data = pd.read_csv(cleaned_posts_path, infer_datetime_format=True)
    one_post_per_day_df = wsb_posts_data.groupby('datetime').last()
    subreddit_subscribers = one_post_per_day_df['subreddit_subscribers'].tolist()

    date_value, date_counts = extract_values_counts_as_lists(wsb_posts_data, 'datetime', False)

    date_post_counts_matching_gme = []
    subreddit_subscribers_matching_gme = []  # for storing the data where the counts match the gme price evolution
    for i, date in enumerate(date_value):
        if date in gme_dates_as_string:
            date_post_counts_matching_gme.append(date_counts[i])
            subreddit_subscribers_matching_gme.append(subreddit_subscribers[i])

    gme_price_history['number_of_posts'] = date_post_counts_matching_gme
    gme_price_history['subreddit_subscribers'] = subreddit_subscribers_matching_gme

    plot_two_df_columns_together(gme_price_history, first_column="Close", second_column="Volume",
                                 third_column="number_of_posts", kind="area",
                                 rescale=True, title="Rescaled Closing Price, Volume, Number of Posts")
