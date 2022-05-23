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


if __name__ == '__main__':
    sns.set_style("darkgrid")
    gme_ticker = "GME"
    gme = yf.Ticker(gme_ticker)
    gme_price_history = get_price_history(gme, "2020-12-01", "2021-02-24")
    plot_closing_price_and_volume(gme_price_history)
    gme_institutional_holders = gme.institutional_holders
    print(gme_institutional_holders)
