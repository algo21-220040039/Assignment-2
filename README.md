# Assignment-2
This assignment seeks to implement support vector machine method to predict whether a stock is likely to increase or decrease in value. The modelling specifications are similar to those in the report added in this repository. Though due to data access problems, this assignment is limited to only stocks in the zhongzheng500 index and the features used are far more restricted.

## Stock pool and features
This assignment only use stocks in the Zhongzheng 500 index, the time range is from 2010/7/1 to 2020/12/31. The first 60 months are used to train the support vector machines and the rest are used to test the prediction and create a portfolio. Due to lack of data access, this project only uses 12 features. Correspondacne between csv file column names and factors used are given below:
| column name | factor|
|:----:|:----|
|turn | turnover rate|
|roeAvg| average of return on equity|
|npMargin| net profit margin|
|gpMargin| gross profit margin|
|YOYAsset|total asset growth rate year on year|
|YOYNI| net profit growth rate year on year|
|currentRatio| current ratio|
|cashRatio| cash ratio|
|liabilityToAsset| liability to asset|
|CFOToNP| operating net cash flow divided by net profit|
|logprice| natural logarithm of close price|
|marketvalue| natural logarithm of market value|

## preprocessing
The collected data underwent the following procedures:
1. winsorization: values outside the range of 5 times the median absolute deviation of median are discarded
2. na filling: na values are filled according to average of corresponding features of companies in the same industry
3. neutralization: First run OLS of feature with respect to industry and market value, then use the residual to replace initial values
4. normalization: use Z-socre to normalize training and test data
5. labeling: average return in the next 30 natural days are used as indicators of stock movement, those ranks among top 30% are labeled 1 while those in the bottom 30% are labeled 0, the rest are dropped.

## svm parameter tuning 
This project uses support vector machine with Gaussian kernel. There are two hyperparameters: C and gamma to select. C is the penalty coefficient and gamma is the spread of kernel funciton. Grid search implementation in scikt_learn is used to determine a combination of C and gamma. The optimial parameters are C=10, gamma=0.01 with score 0.52

## performance on the test data
Two metrics are used to determine the performance, namely accuracy and AUC

## constructing a portfolio
In the testing month, 50 stocks with highest score are selected with value averaged weights, the return are plotted as below
