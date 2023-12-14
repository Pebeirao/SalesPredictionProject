from typing import List

from pandas import DataFrame


def create_day_column(sales_calendar: DataFrame) -> List[str]:
    """Create a day column for the sales_calendar dataframe"""
    sales_calendar.sort_values(by=["date"], inplace=True)

    return "d_" + (sales_calendar.index + 1).astype(str)


def summarise_sales_train_validation(sales_train_validation: DataFrame) -> DataFrame:
    sales_train_validation = sales_train_validation.drop(columns=["id"])
    sales_train_validation = sales_train_validation.sum()

    sales_train_validation = sales_train_validation.reset_index()
    return sales_train_validation.rename(columns={"index": "day", 0: "sales"})
