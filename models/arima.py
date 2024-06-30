import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima.model import ARIMA


def sliding_window(data, window_size):
    windows = []
    for i in range(len(data) - window_size + 1):
        window = data.iloc[i:i + window_size]
        windows.append(window)
    return windows


if __name__ == "__main__":
    turbine_data = {}
    grouped_data = pd.read_csv("raw_data/time_series_simple.csv").groupby("TurbID")

    for group_name, group_df in grouped_data:
        # set window size to a day
        window_size = 143
        turbine_data[group_name] = sliding_window(group_df, window_size)

    # show autocorrelation for first day
    example_serie = turbine_data[1][0]['Patv']

    autocorrelation_plot(example_serie)
    plt.show()

    # fit model: 5 lags autoreg, 1st order diff, no moving average
    model = ARIMA(example_serie, order=(5, 1, 0))
    model_fit = model.fit()
    # summary of fit model
    print(model_fit.summary())
    # line plot of residuals
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    plt.show()
    # density plot of residuals
    residuals.plot(kind='kde')
    plt.show()
    # summary stats of residuals
    print(residuals.describe())
