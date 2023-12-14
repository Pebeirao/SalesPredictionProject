from enum import Enum
from typing import Union

from pandas import DataFrame


class ArimaModels(Enum):
    ARIMA = "ARIMA"
    SARIMA = "SARIMA"
    SARIMAX = "SARIMAX"

    def __str__(self):
        return self.value

    def model_frequency(self) -> int:
        if self == ArimaModels.ARIMA:
            return 1
        elif self == ArimaModels.SARIMA:
            return 12
        elif self == ArimaModels.SARIMAX:
            return 12
        else:
            raise ValueError()

    def model_seasonal(self) -> bool:
        if self == ArimaModels.ARIMA:
            return False
        elif self == ArimaModels.SARIMA:
            return True
        elif self == ArimaModels.SARIMAX:
            return True
        else:
            raise ValueError()

    def model_exogenous(
        self, data_frame: Union[DataFrame, None]
    ) -> Union[DataFrame, None]:
        if self == ArimaModels.ARIMA:
            return None
        elif self == ArimaModels.SARIMA:
            return None
        elif self == ArimaModels.SARIMAX:
            return data_frame
        else:
            raise ValueError()
