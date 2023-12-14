import pathlib

import matplotlib.pylab as plt
import pandas as pd
import pmdarima as pm
from pandas import DataFrame, date_range, DateOffset

from sales_prediction import ROOT_DIR
from sales_prediction.constants.arimaConstants import ArimaModels
from sales_prediction.constants.offsetconstants import OffsetEnum


class ArimaTraining:
    """Class with the Main ARIMA Models"""

    def __init__(
        self,
        train_data: DataFrame,
        model: ArimaModels,
        exogenous_data: DataFrame = None,
    ):
        self.predicted_column = None
        self.periods = None
        self.confidence = None
        self.fitted = None
        self.model_trained = None
        self.train_data = train_data
        self.exogenous_data = exogenous_data
        self.model = model

    def train_model(
        self,
        predicted_column: str,
    ):
        """Standard ARIMA Model"""
        self.predicted_column = predicted_column

        self.model_trained = pm.auto_arima(
            self.train_data[predicted_column],
            exogenous=self.model.model_exogenous(self.exogenous_data),
            start_p=1,
            start_q=1,
            test="adf",  # use adftest to find optimal 'd'
            max_p=3,
            max_q=3,  # maximum p and q
            m=self.model.model_frequency(),  # frequency of series (if m==1, seasonal is set to FALSE automatically)
            d=None,  # let model determine 'd'
            seasonal=self.model.model_seasonal(),  # No Seasonality for standard ARIMA
            trace=False,  # logs
            error_action="warn",  # shows errors ('ignore' silences these)
            suppress_warnings=True,
            stepwise=True,
        )

    def forecast(self, periods: int = 24):
        """Forecast the next n periods"""
        self.periods = periods
        fitted, confidence = self.model_trained.predict(
            n_periods=periods, return_conf_int=True
        )
        self.fitted = fitted
        self.confidence = confidence

    def plot_forecast(self, frequency: OffsetEnum = OffsetEnum.D):
        index_of_fc = date_range(
            DateOffset().rollforward(dt=self.train_data.index[-1]),
            periods=self.periods,
            freq=str(frequency),
        )

        # make series for plotting purpose
        fitted_series = pd.Series(self.fitted, index=index_of_fc)
        lower_series = pd.Series(self.confidence[:, 0], index=index_of_fc)
        upper_series = pd.Series(self.confidence[:, 1], index=index_of_fc)

        # Plot
        plt.figure(figsize=(15, 7))
        plt.plot(self.train_data[self.predicted_column], color="#1f76b4")
        plt.plot(fitted_series, color="darkgreen")
        plt.fill_between(
            lower_series.index, lower_series, upper_series, color="k", alpha=0.15
        )

        plt.title(
            f"{self.model} - Forecast of {self.predicted_column} Per {str(frequency)}"
        )
        plt.savefig(
            pathlib.Path(
                ROOT_DIR,
                "images",
                f"{self.model}_{self.predicted_column}_forecast.png",
            )
        )
        plt.show()
