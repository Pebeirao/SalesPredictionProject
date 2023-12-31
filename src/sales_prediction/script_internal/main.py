import pathlib

from pandas import read_csv, merge, to_datetime

from sales_prediction import ROOT_DIR
from sales_prediction.classes.arima.arimaMain import ArimaTraining
from sales_prediction.constants.arimaConstants import ArimaModels
from sales_prediction.script_internal.transformation import (
    create_day_column,
    summarise_sales_train_validation,
)


def main():
    sales_calendar = read_csv(
        pathlib.Path(
            ROOT_DIR,
            "data",
            "raw",
            "calendar_afcs2023.csv",
        )
    )
    sales_train_validation = read_csv(
        pathlib.Path(
            ROOT_DIR,
            "data",
            "raw",
            "sales_train_validation_afcs2023.csv",
        )
    )

    # add day variable that is a d_ plus 1 to the end of the calendar
    sales_calendar["date"] = to_datetime(sales_calendar["date"], format="%m/%d/%Y")
    sales_calendar["day"] = create_day_column(sales_calendar)

    # summarise the sales per day by summing the sales of each item
    sales_by_day = summarise_sales_train_validation(sales_train_validation)

    # merge the two dataframes
    sales_byday_with_date = merge(sales_by_day, sales_calendar, on="day")

    # set the index to be the date
    sales_byday_with_date = sales_byday_with_date.set_index(["date"])
    
    # print(sales_byday_with_date.loc[:, 'event_name_1':'event_type_2'].fillna(0).head())

    # arima_fitted = ArimaTraining(sales_byday_with_date, ArimaModels.ARIMA)
    # arima_fitted.train_model("sales")
    # arima_fitted.forecast(periods=28)
    # arima_fitted.plot_forecast()

    # sarima_fitted = ArimaTraining(sales_byday_with_date, ArimaModels.SARIMA)
    # sarima_fitted.train_model("sales")
    # sarima_fitted.forecast(periods=28)
    # sarima_fitted.plot_forecast()

    sarimax_fitted = ArimaTraining(
        sales_byday_with_date, ArimaModels.SARIMAX, sales_byday_with_date.loc[:, 'event_name_1':'event_type_2'].fillna(0)
    )
    sarimax_fitted.train_model("sales")
    sarimax_fitted.forecast(periods=28)
    sarimax_fitted.plot_forecast()


def arima():
    sales_calendar = read_csv(
        pathlib.Path(
            ROOT_DIR,
            "data",
            "raw",
            "calendar_afcs2023.csv",
        )
    )
    sales_train_validation = read_csv(
        pathlib.Path(
            ROOT_DIR,
            "data",
            "raw",
            "sales_train_validation_afcs2023.csv",
        )
    )

    # add day variable that is a d_ plus 1 to the end of the calendar
    sales_calendar["date"] = to_datetime(sales_calendar["date"], format="%m/%d/%Y")
    sales_calendar["day"] = create_day_column(sales_calendar)

    # summarise the sales per day by summing the sales of each item
    sales_by_day = summarise_sales_train_validation(sales_train_validation)

    # merge the two dataframes
    sales_byday_with_date = merge(sales_by_day, sales_calendar, on="day")

    # set the index to be the date
    sales_byday_with_date = sales_byday_with_date.set_index(["date"])

    arima_fitted = ArimaTraining(sales_byday_with_date, ArimaModels.ARIMA)
    arima_fitted.train_model("sales")
    arima_fitted.forecast(periods=300)
    arima_fitted.plot_forecast()


def sarima():
    sales_calendar = read_csv(
        pathlib.Path(
            ROOT_DIR,
            "data",
            "raw",
            "calendar_afcs2023.csv",
        )
    )
    sales_train_validation = read_csv(
        pathlib.Path(
            ROOT_DIR,
            "data",
            "raw",
            "sales_train_validation_afcs2023.csv",
        )
    )

    # add day variable that is a d_ plus 1 to the end of the calendar
    sales_calendar["date"] = to_datetime(sales_calendar["date"], format="%m/%d/%Y")
    sales_calendar["day"] = create_day_column(sales_calendar)

    # summarise the sales per day by summing the sales of each item
    sales_by_day = summarise_sales_train_validation(sales_train_validation)

    # merge the two dataframes
    sales_byday_with_date = merge(sales_by_day, sales_calendar, on="day")

    # set the index to be the date
    sales_byday_with_date = sales_byday_with_date.set_index(["date"])

    sarima_fitted = ArimaTraining(sales_byday_with_date, ArimaModels.SARIMA)
    sarima_fitted.train_model("sales")
    sarima_fitted.forecast(periods=300)
    sarima_fitted.plot_forecast()


def sarimax():
    sales_calendar = read_csv(
        pathlib.Path(
            ROOT_DIR,
            "data",
            "raw",
            "calendar_afcs2023.csv",
        )
    )
    sales_train_validation = read_csv(
        pathlib.Path(
            ROOT_DIR,
            "data",
            "raw",
            "sales_train_validation_afcs2023.csv",
        )
    )

    # add day variable that is a d_ plus 1 to the end of the calendar
    sales_calendar["date"] = to_datetime(sales_calendar["date"], format="%m/%d/%Y")
    sales_calendar["day"] = create_day_column(sales_calendar)

    # summarise the sales per day by summing the sales of each item
    sales_by_day = summarise_sales_train_validation(sales_train_validation)

    # merge the two dataframes
    sales_byday_with_date = merge(sales_by_day, sales_calendar, on="day")

    # set the index to be the date
    sales_byday_with_date = sales_byday_with_date.set_index(["date"])

    sarimax_fitted = ArimaTraining(
        sales_byday_with_date, ArimaModels.SARIMAX, sales_byday_with_date["wday"]
    )
    sarimax_fitted.train_model("sales")
    sarimax_fitted.forecast(periods=300)
    sarimax_fitted.plot_forecast()


if __name__ == "__main__":
    main()
