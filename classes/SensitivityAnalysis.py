import numpy as np


def calculate_mse(simulation_price_history, gme_price_history):
    mse = np.square(np.subtract(simulation_price_history, gme_price_history))
    return mse
