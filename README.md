# Forecasting of Forex Market Based on Machine Learning Algorithms

## Introduction

The project explains how we can apply machine learning algorithms for forex market trend prediction with the help of past market data. The algorithms implemented in this project are ARIMA, Random Forest, RNN, and LSTM. The algorithms utilized here consume the past market data (500 days) of EURUSD and make use of technical indicators like moving averages, Relative Strength Indicator (RSI), Bollinger Bands, and others to teach our models.

## Data Sources

**OANDA**: Foreign exchange market real-time data collected via API.  
The data contains:  
- **time**: Timestamp of each price action (30-minute intervals).  
- **open, high, low, close**: Prices for each interval.  
- **volume**: The volume traded during the interval.

## Prediction Objective

The goal is to provide predictions of Buy/Sell signals for the forex market. These predictions will help to inform trading decisions in real-time or aid in automated trading systems so that we can make calculated risks based on market states.

## Process Overview

We initially downloaded a dataset with identical structure to OANDA forex data but realized it did not provide real-time updates. Hence, we decided to use OANDA forex data only and applied feature engineering techniques on (open, high, low, close, volume, time). To improve the accuracy of predictions, macroeconomic data from FRED; however, OANDA updates in 30-minute intervals, while FRED updates quarterly/yearly. So, we decided to drop the extra dataset as it was adding less variance to our data.

Aside from the incompatibility between the two datasets, we also aim to focus on swing trading in this project, which requires short to medium-term price movements, something that the infrequently updated FRED data lacks.

Our dataset also showed imbalancement in target classes (Buy/Sell/Hold), influenced by the daily volatility of incoming live data. To address this, we applied RandomOverSampling to balance the dataset. We have used four machine learning models in this project to build our model: ARIMA, Random Forest, RNN, and LSTM. To optimize model performance, we used GridSearchCV, RandomizedSearchCV, and Keras' built-in tools to find the best hyperparameters.

## Exploratory Data Analysis (EDA)

- **Number of Observations**: 500  
- **X and Y Variables**:  
    - **X**: Features extracted from the dataset (open, high, low, close, volume) along with engineered indicators such as MACD, RSI, future return, volatility bands, and other trend indicator signals.  
    - **Y**: Target variable representing trading signals — Buy (1), Sell (2), Hold (0).  

![Price Vs Predictors](figures/priceVsPredictors.png)

### Target Data Distribution

![Target Data Distribution](figures/TargetDistribution.png)

### Feature Correlation

![Feature correlation matrix](figures/correlation.png)

## Model Selection

The task is a classification problem, where we predict discrete classes (Buy/Sell/Hold).

### Model Fitting

**Train-Test Split and Data Leakage Risk**:  
To make a realistic evaluation, the dataset was chronologically ordered and split into three distinct segments:  
- **Training Data**: 70% of the earliest historical data  
- **Validation Data**: 15% of the mid-range data  
- **Test Data**: Final 15% representing the most recent data  

This time-based splitting approach avoids lookahead bias that would have been caused due to the 30-minute interval records, ensuring that future values are not leaked into the training process. The feature data was prepared by dropping non-numeric and identifier columns *(date_only, time, signal)* and separating the target variable *signal*, which represents Buy, Sell, or Hold actions.

To address class imbalance, RandomOverSampler from imblearn was applied to the training set. This step resampled the data to ensure that each class (Buy, Sell, Hold) was represented more evenly, reducing bias toward the dominant class. Finally, we used a MinMaxScaler to normalize the feature values. The scaler was fitted only on the resampled training data and then applied to the validation and test data.

### Model Choice

The following models were used to predict forex signals:

#### ARIMA

The Autoregressive Integrated Moving Average (ARIMA) is well-suited for forecasting continuous time series data. In our case, to align with ARIMA's requirements, we modified the target variable from the categorical signal (Buy, Sell, Hold) to a continuous variable, future_return. This allows ARIMA to forecast future returns, which can then be interpreted indirectly to inform Buy, Sell, or Hold decisions based on thresholds or strategy rules.  


**Confusion Matrix**
![Confusion Matrix](figures/ARIMAConfusion.png)

**Line Graph for Actual and Predicted Values**  
![Actual Vs Predicted](figures/ARIMALineGraph.png)

**Classification Report**

| Class   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Hold    | 0.65      | 0.83   | 0.73     | 42      |
| Buy     | 0.00      | 0.00   | 0.00     | 13      |
| Sell    | 0.29      | 0.18   | 0.22     | 11      |
|         |           |        |          |         |
| **Accuracy**      |           |        | **0.56**     | **66**     |
| **Macro Avg**     | 0.31      | 0.34   | 0.32     | 66      |
| **Weighted Avg**  | 0.46      | 0.56   | 0.50     | 66      |


#### Ensemble Approach

**Random Forest**  
Random Forest is used to classify Buy, Sell, or Hold signals based on a variety of features. We believed it will do a better classification by capturing complex relationships between input features and target signals, while also handling feature importance efficiently.  


**Confusion Matrix**
![Confusion Matrix](figures/RandomForestConfusion.png)

**Line Graph for Actual and Predicted Values**  
![Actual Vs Predicted](figures/RandomForestLineGraph.png)

**Classification Report**

| Class            | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| 0                | 1.00      | 0.75   | 0.86     | 4       |
| 1                | 0.91      | 0.74   | 0.82     | 39      |
| 2                | 0.69      | 0.92   | 0.79     | 24      |
|                  |           |        |          |         |
| **Accuracy**     |           |        | **0.81** | **67**  |
| **Macro Avg**    | 0.86      | 0.80   | 0.82     | 67      |
| **Weighted Avg** | 0.83      | 0.81   | 0.81     | 67      |


#### Neural Network Approach
**LSTM**  
Long Short-Term Memory (LSTM) networks are an advanced type of RNN designed to capture both short and long-term dependencies in sequential data. LSTMs were applied for their ability to model complex temporal patterns in forex market trends more effectively.  


**Confusion Matrix**
![Confusion Matrix](figures/LSTMconfusion.png)

**Line Graph for Actual and Predicted Values**  
![Actual Vs Predicted](figures/LstmLinegraph.png)


**Classification Report**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Hold  | 0.20      | 0.50   | 0.29     | 4       |
| Buy   | 0.78      | 0.92   | 0.85     | 39      |
| Sell  | 0.91      | 0.42   | 0.57     | 24      |
|       |           |        |          |         |
| **Accuracy**      |         |        | **0.72** | **67** |
| **Macro Avg**     | 0.63    | 0.61   | 0.57     | 67      |
| **Weighted Avg**  | 0.79    | 0.72   | 0.71     | 67      |


## Observation and Conclusion

## Future Work
**RNN**  
Recurrent Neural Networks (RNNs) are designed to capture temporal dependencies in sequential data. We used the RNN model to capture short-term patterns in forex market data by retaining information from previous time steps, making them well-suited for time series forecasting and signal classification.  


**Confusion Matrix**
![Confusion Matrix](figures/RNNConfusion.png)

**Line Graph for Actual and Predicted Values**  
![Actual Vs Predicted](figures/RNNLineGraph.png)


**Classification Report**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Hold  | 0.50      | 0.75   | 0.60     | 4       |
| Buy   | 0.78      | 0.92   | 0.85     | 39      |
| Sell  | 0.87      | 0.54   | 0.67     | 24      |
|       |           |        |          |         |
| **Accuracy**      |         |        | **0.78** | **67** |
| **Macro Avg**     | 0.72    | 0.74   | 0.70     | 67      |
| **Weighted Avg**  | 0.80    | 0.78   | 0.77     | 67      |

Observation and Conclusion

Future Work
