import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from classes.MarketEnvironment import MarketEnvironment
from classes.SimulationClass import SimulationClass, start_simulation
from helpers.calculations_helpers import rescale_array
from helpers.plotting_helpers import get_price_history, select_closing_prices

sns.set_style('darkgrid')

class ParameterEstimation:
    def __init__(self):
        self.commitment_histories = []
        self.deicision_dictionaries = []

    def estimate_mu_parameter(self, gme_price_history):
        mu_parameters = np.linspace(0.1, 1, 20)
        for mu in mu_parameters:
            prices, aaa, bbb, average_commitment_history, _ = start_simulation(miu=mu)
            self.commitment_histories.append(average_commitment_history)
        self.plot_mu_calibration(mu_parameters, gme_price_history)

    def plot_mu_calibration(self, mu_params, gme_price_history):
        plt.figure(figsize=(8, 8))
        legend_entries = [f'mu = {mu}' for mu in mu_params]
        for history in self.commitment_histories:
            history = rescale_array(history)
            plt.plot(history[:50])
        gme_price_history = rescale_array(gme_price_history)
        plt.plot(gme_price_history[:50], "bx")
        plt.grid(True)
        legend_entries.append("GME Price")
        plt.legend(legend_entries)
        plt.xlabel("Simulation Day", fontsize=18)
        plt.ylabel("Average Commitment Evolution", fontsize=18)
        plt.title(r"Parameter Estimation of $\mu$", fontsize=20)
        plt.show()


    def estimate_lambda(self):
        lambdas = np.linspace(1, 2.5, 20)
        for lambda_ in lambdas:
            prices, aaa, bbb, _, hedge_fund_decision_dict = start_simulation(lambda_parameter=lambda_)
            self.deicision_dictionaries.append(hedge_fund_decision_dict)
        self.plot_lambda_calibration(lambdas)

    def plot_lambda_calibration(self, lambdas):
        plt.figure(figsize=(10, 12))
        legend_entries = [f'lambda = {lambda_}' for lambda_ in lambdas]
        for dictionary in self.deicision_dictionaries:
            short_gme_decisions = []
            close_position_decisions = []
            for key, decision in dictionary.items():
                short_gme_decisions.append(decision.count(True))
                close_position_decisions.append(decision.count(False))
            plt.plot(short_gme_decisions)
        plt.xlabel("Simulation Day", fontsize=18)
        plt.ylabel("Institutional Investor Decisions to short GME throughout simulation", fontsize=18)
        plt.title(r"Prameter estimation of $\lambda$", fontsize=20)

        plt.legend(legend_entries, fontsize=14)
        plt.show()

if __name__ == '__main__':
    start_date = datetime.datetime(2020, 12, 7)
    gme = yf.Ticker("GME")
    gme_price_history = get_price_history(gme, "2020-11-15", "2020-12-08")
    gme_price_history = select_closing_prices(gme_price_history)
    market_environment = MarketEnvironment(initial_price=16.35, name="Parameter Estimation Environment",
                                           price_history=gme_price_history, start_date=start_date)
    simulation = SimulationClass(time_steps=100, N_agents=10000, N_institutional_investors=200, m=4,
                                 market_environment=market_environment, miu=0.17, commitment_scaler=1.5,
                                 volume_threshold=0.97, fundamental_price_inst_inv=1, lambda_parameter=1.75)
    estimation = ParameterEstimation()
    gme_price_history = get_price_history(gme, "2020-11-15", "2021-02-08")
    gme_price_history = select_closing_prices(gme_price_history)
    estimation.estimate_lambda()