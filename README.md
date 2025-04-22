- Data sources are:
  - OANDA : forex market data Collected via API, updated in real-time based on forex market transactions. We have columns like:
        - time – Timestamp of each price movement (30-minute intervals). 
        - open, high, low, close – Price levels for the given timeframe.
          - volume – Total trading volume within the period.
  - FRED (Federal Reserve Economic Data): For macroeconomics data  regarding:
        - GDP Growth Rate (series_id=GDPC1)
        - Unemployment Rate (series_id=UNRATE)
        - Interest Rates (10-Year Treasury) (series_id=DGS10)
        - Consumer Price Index (Inflation) (series_id=CPIAUCSL)
  - Economic News Sentiment (BBC RSS): for fetching real time economic news data that would affect the market. It was collected using an API.
      - headline :  News article titles related to macroeconomic trends. keywords like 'interest rate', 'inflation', 'central bank', 'monetary policy'.
    
- Outline what you plan to predict.  How might this prediction be used in production or in practice?
    - key market actions:  Buy/Sell signals 

- Process overview 
    - Initially, we downloaded a dataset with the same columns as the OANDA forex dataset but realized it did not provide real-time updates, only data up to the download timestamp. Since our project required the most recent market data, we decided to rely on OANDA forex data alone for predictions, applying feature engineering on OHLCV (open, high, low, close, volume, time).
    To improve trading accuracy, we later integrated macroeconomic indicators from FRED and news sentiment from BBC RSS.
    - During this process, we encountered timestamp mismatches:
        - OANDA updates every 30 minutes, while FRED updates quarterly/yearly, leading to many NaN values.
        - To resolve this, we resampled FRED data to daily intervals, ensuring macroeconomic trends aligned better with forex price movements.
- EDA 
  - What are your X and Y variables?
      - X is all the feature in the dataset 
      - Y is Signal (Buy, Sell, Hold)

- Classification or regression?  
    - Classification

- How many observations?
    - 500

- What is the distribution of each feature?  You don't need to show every feature's distribution. 
    - There are more hold signals compared to buy or sell. We will need to balance the target labels if necessary. 
    - Macroeconomic Indicators have low variance compared to the high-frequency forex price movements because they are updated quarterly/yearly. This could make predictions too dependent on outdated macro trends. 

- Correlation - are some features strongly correlated?  

- Feature importance.
    - We are using features we have engineered to get these technical indicators, We can use them to capture trend, volatility, and price momentum (EMA, RSI, MACD, Bollinger Bands, ATR)
    - Macroeconomic factors in the market (CPIAUCSL, GDPC1, UNRATE, DGS10).
    -  And news-driven market reactions from the headlines

- Feature engineering
  - Which features needed feature engineering? 
      - Feature engineering was done to transform raw forex and macroeconomic data into meaningful insights for predictions. We will be using most of the transformed data.

- Model fitting
    - Train / test splitting
      - How was this performed? How did you decide the train/test sizes?

- Does your dataset have a risk of data leakage? Describe those risks.

    - Risk of timestamp misalignment since we are merging a dataset which updates every 30 minutes, a dataset that updates quarterly/yearly and a dataset   that is updated as soon as anything related to the keywords is mentioned.

- Which model did you select, why?
    - XGBoost Classifier: it is used to make time-based predictions based on historical data and it does better.  

- What was the process for hyper parameter selection if applicable.  
    - GridSeachCV was used to select hyperparameters. 

- Validation / metrics
    - Which metrics did you weigh most heavily, why?
        - Accuracy, Recall and r^2
            - Confusion matrix and confusion discussion for classification problems
        -  Highlight model weaknesses
        - Give 2-4 prediction examples from your data.
        - Give 2 prediction examples which are new or synthesized. 

- Overfitting/Underfitting
    - We are using  grid search hyperparameters  in order to mitigate overfitting. We will be doing more accuracy tests between the training and test splits to see if the model is overfitting or underfitting. 
- Production  
    ?

- Going further we will work on 
    - finding access to a real time macroeconomics data  to improve trade accuracy,   
    - Make better prediction test cases.


