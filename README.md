# Technical Description

## Start the project with the following command:
```
pip install -e .
```
This will install the package in editable mode, meaning that changes to the source code will be immediately available to other users of the package on your system.

*Note: You may need to use pip3 instead of pip depending on your system configuration.*

## Run the project with the following command:
```
train_models
```
#### or
```
python3 main.py
```

### Also u can run one Model with the following command:

```
train_arima
```

#### or

```
train_sarima
```

#### or

```
train_sarimax
```
## Models can be find at the folder
src/sales_prediction/classes/arima

# The Task:

Forecasting predicts the number of sales in the future. Having the right amount of products in stock is a core challenge in retail. A good forecast makes sure there are enough of your favourite products in stock, even if you come to the store late in the evening.

In this project, you will use a subset of M5 Forecasting - Accuracy hierarchical sales data from Walmart at one store, TX3 in the State of Texas, the world’s largest company by revenue, to forecast daily sales for the next 28 days. The data include item level, department, product categories, and store details. In addition, it has explanatory variables such as price, promotions, day of the week, and special events. Altogether, it can be used to improve forecasting accuracy. 

*Final Project deadline 21/12 23:59*

# The Dataset: 

The subset of M5 dataset, generously made available by Walmart, involves the unit sales of various products sold in the USA, more specifically, the dataset involves the unit sales of 3,049 products, classified into 3 product categories (Hobbies, Foods, and Household) and 7 product departments, in which the above-mentioned categories are disaggregated.

For this project, the selected products, Food3, are sold by TX3 store, located in Texas. 

The dataset consists of the following five (5) files:

## File 1: “calendar_afcs2023.csv” contains information about the dates the products are sold.

- **date**: The date in a “y-m-d” format.
- **wm_yr_wk**: The id of the week the date belongs to.
- **weekday**: The type of the day (Saturday, Sunday, …, Friday).
- **wday**: The id of the weekday, starting from Saturday.
- **month**: The month of the date.
- **year**: The year of the date.
- **event_name_1**: If the date includes an event, the name of this event.
- **event_type_1**: If the date includes an event, the type of this event.
- **event_name_2**: If the date includes a second event, the name of this event.
- **event_type_2**: If the date includes a second event, the type of this event.
- **snap_TX**: A binary variable (0 or 1) indicating whether the stores of TX, allow SNAP3 purchases on the examined date. 1 indicates that SNAP purchases are allowed.
 

## File 2: “sell_prices_afcs2023.csv” Contains information about the price of the products sold per store and date.

- **store_id**: The id of the store where the product is sold.
- item_id: The id of the product.
- wm_yr_wk: The id of the week.
- sell_price: The price of the product for the given week/store. The price is provided per week (average across seven days). If not available, this means that the product was not sold during the examined week. Note that although prices are constant at weekly basis, they may change through time (both training and test set).

## File 3: “sales_train_validation_afcs2023.csv Download sales_train_validation_afcs2023.csv”, contains the historical daily unit sales data per product and store.

- item_id: The id of the product.
- dept_id: The id of the department the product belongs to.
- cat_id: The id of the category the product belongs to.
- store_id: The id of the store where the product is sold.
- state_id: The State where the store is located.
- d_1, d_2, …, d_i, … d_1913: The number of units sold at day i, starting from 2011-01-29.

## File 4: “sales_test_validation_afcs2023.csv”, contains the historical daily unit sales data per product and store.
- **item_id**: The id of the product. 
- **dept_id**: The id of the department the product belongs to.
- **cat_id**: The id of the category the product belongs to.
- **store_id**: The id of the store where the product is sold.
- **state_id**: The State where the store is located.
- **d_1914**, **d_1925**, …, **d_i**, … **d_1941**: The number of units sold at day i, starting from 2017-01-29

## File 5: "sample_submission_afcs2023.csv" contains the number of forecasts to be submitted for point forecasts, exactly 28 days (4 weeks ahead), starting at F1, F2, …, F28.