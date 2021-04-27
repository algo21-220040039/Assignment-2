import pandas as pd
import numpy as np
import baostock as bs
import glob
from scipy import stats

# get the stock listed in zhongzheng500
lg = bs.login()
rs = bs.query_zz500_stocks()
zz500_stocks = []
while (rs.error_code == '0') & rs.next():
    zz500_stocks.append(rs.get_row_data())
result = pd.DataFrame(zz500_stocks, columns=rs.fields)
result.to_csv("D:/zz500_stocks.csv", encoding="gbk", index=False)
bs.logout()

zz500 = pd.read_csv('D:/zz500_stocks.csv', encoding='gbk')
zz500_stock_list = list(zz500['code'])

# daily data for all stocks in zz500
for stock in zz500_stock_list:
    lg = bs.login()
    rs = bs.query_history_k_data_plus(stock,
                                      "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
                                      start_date='2010-01-01', end_date='2020-12-31',
                                      frequency="d", adjustflag="2")
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    result.to_csv(r"D:\stockdata\%s.csv"%stock, encoding="gbk", index=False)
    bs.logout()

path =r'D:/test' # use your path
big_frame = pd.concat([pd.read_csv(f, encoding='gbk') for f in glob.glob(path + "/*.csv")],
                      ignore_index=True)

# request other accounting data for constructing factors
for stock in zz500_stock_list:
    result = pd.DataFrame()
    for year in range(2010, 2021):
        for quarter in range(1, 5):
            bs.login()
            profit_list = []
            rs_profit = bs.query_profit_data(code=stock, year=year, quarter=quarter)
            while (rs_profit.error_code == '0') & rs_profit.next():
                profit_list.append(rs_profit.get_row_data())
            result_profit = pd.DataFrame(profit_list, columns=rs_profit.fields)
            bs.logout()
            result = pd.concat([result, result_profit], ignore_index=True)
    result.to_csv(r"D:\profitdata\%s.csv"%stock, encoding="gbk", index=False)

for stock in zz500_stock_list:
    result = pd.DataFrame()
    for year in range(2010, 2021):
        for quarter in range(1, 5):
            bs.login()
            profit_list = []
            operation_list = []
            rs_operation = bs.query_operation_data(code=stock, year=year, quarter=quarter)
            while (rs_operation.error_code == '0') & rs_operation.next():
                operation_list.append(rs_operation.get_row_data())
            result_operation = pd.DataFrame(operation_list, columns=rs_operation.fields)
            bs.logout()
            result = pd.concat([result, result_operation], ignore_index=True)
    result.to_csv(r"D:\operationdata\%s.csv"%stock, encoding="gbk", index=False)

for stock in zz500_stock_list:
    result = pd.DataFrame()
    for year in range(2010, 2021):
        for quarter in range(1, 5):
            bs.login()
            growth_list = []
            rs_growth = bs.query_growth_data(code=stock, year=year, quarter=quarter)
            while (rs_growth.error_code == '0') & rs_growth.next():
                growth_list.append(rs_growth.get_row_data())
            result_growth = pd.DataFrame(growth_list, columns=rs_growth.fields)
            bs.logout()
            result = pd.concat([result, result_growth], ignore_index=True)
    result.to_csv(r"D:\growthdata\%s.csv"%stock, encoding="gbk", index=False)

for stock in zz500_stock_list:
    result = pd.DataFrame()
    for year in range(2010, 2021):
        for quarter in range(1, 5):
            bs.login()
            growth_list = []
            rs_balance = bs.query_balance_data(code=stock, year=year, quarter=quarter)
            while (rs_balance.error_code == '0') & rs_balance.next():
                growth_list.append(rs_balance.get_row_data())
            result_balance = pd.DataFrame(growth_list, columns=rs_balance.fields)
            bs.logout()
            result = pd.concat([result, result_balance], ignore_index=True)
    result.to_csv(r"D:\balancedata\%s.csv"%stock, encoding="gbk", index=False)

for stock in zz500_stock_list:
    result = pd.DataFrame()
    for year in range(2010, 2021):
        for quarter in range(1, 5):
            bs.login()
            growth_list = []
            rs_cash_flow = bs.query_cash_flow_data(code=stock, year=year, quarter=quarter)
            while (rs_cash_flow.error_code == '0') & rs_cash_flow.next():
                growth_list.append(rs_cash_flow.get_row_data())
            result_cashflow = pd.DataFrame(growth_list, columns=rs_cash_flow.fields)
            bs.logout()
            result = pd.concat([result, result_cashflow], ignore_index=True)
    result.to_csv(r"D:\cashflowdata\%s.csv"%stock, encoding="gbk", index=False)

data = pd.read_csv('D:/stockdata/sh.600006.csv', encoding='gbk')
data['date'] = pd.to_datetime(data['date'], format='%Y/%m/%d')
curr_month = data.iloc[0,0].month
start = 0
target_date = []
for row in range(1, data.shape[0]):
    if data.iloc[row, 0].month != curr_month:
        curr_month = data.iloc[row, 0].month
        target_date.append(data.iloc[(start + row)//2, 0])
        start = row
target_date.append(data.iloc[(start + data.shape[0])//2, 0])

# query csi300 index
lg = bs.login()
rs = bs.query_history_k_data_plus("sh.000300",
    "date,code,open,high,low,close,preclose,volume,amount,pctChg",
    start_date='2010-01-01', end_date='2020-12-31', frequency="d")
data_list = []
while (rs.error_code == '0') & rs.next():
    data_list.append(rs.get_row_data())
result = pd.DataFrame(data_list, columns=rs.fields)
result.to_csv("D:\\csi300.csv", index=False)
print(result)
bs.logout()

# query industry classification
lg = bs.login()
rs = bs.query_stock_industry()
industry_list = []
while (rs.error_code == '0') & rs.next():
    industry_list.append(rs.get_row_data())
result = pd.DataFrame(industry_list, columns=rs.fields)
result.to_csv("D:/stock_industry.csv", encoding="gbk", index=False)
bs.logout()

result = pd.read_csv('D:/stock_industry.csv', encoding='gbk')
result.astype(str)
industry_dic = {}
for i in range(0, result.shape[0]):
    if result.iloc[i, 1] in zz500_stock_list:
        industry_dic[result.iloc[i, 1]] = result.iloc[i, 3]

# data cleaning and preparation
# csi300 = pd.read_csv(r"D:\csi300.csv", encoding="gbk", usecols= ['date', 'pctChg'])
for stock in zz500_stock_list:
    # read the stock price daily data and compute 30davg return for labelling
    price_col_list = ['date', 'code', 'close', 'turn', 'tradestatus', 'pctChg', 'isST']
    stockdata = pd.read_csv(r"D:\stockdata\%s.csv"%stock, encoding="gbk", usecols=price_col_list)
    stockdata['date'] = pd.to_datetime(stockdata['date'], format='%Y/%m/%d')
    result = []
    for row in range(0, stockdata.shape[0] - 23):
        startdate = stockdata.iloc[row, 0]
        i = 20
        while((stockdata.iloc[row+i, 0] - startdate) < pd.Timedelta(days=30)):
            i = i+1
        result.append(stockdata.iloc[row+1:row+i, -2].mean())
    stockdata['avg30dreturn'] = pd.DataFrame(result)

    profit_col_list = ['pubDate', 'roeAvg', 'gpMargin', 'npMargin', 'totalShare']
    profitdata = pd.read_csv(r'D:/profitdata/%s.csv'%stock, encoding='gbk', usecols=profit_col_list)
    profitdata['pubDate'] = pd.to_datetime(profitdata['pubDate'], format='%Y/%m/%d')
    profitdata.drop_duplicates(subset=['pubDate'], keep='last', inplace=True)
    stockdata = stockdata.merge(profitdata, left_on='date', right_on='pubDate', how='left')
    stockdata.drop(columns=['pubDate'], inplace=True)

    growth_col_list = ['pubDate', 'YOYAsset', 'YOYNI']
    growthdata = pd.read_csv(r'D:/growthdata/%s.csv'%stock, encoding='gbk', usecols=growth_col_list)
    growthdata['pubDate'] = pd.to_datetime(growthdata['pubDate'], format='%Y/%m/%d')
    growthdata.drop_duplicates(subset=['pubDate'], keep='last', inplace=True)
    stockdata = stockdata.merge(growthdata, left_on='date', right_on='pubDate', how='left')
    stockdata.drop(columns=['pubDate'], inplace=True)

    balance_col_list = ['pubDate', 'currentRatio', 'cashRatio', 'liabilityToAsset']
    balancedata = pd.read_csv(r'D:/balancedata/%s.csv' % stock, encoding='gbk', usecols=balance_col_list)
    balancedata['pubDate'] = pd.to_datetime(balancedata['pubDate'], format='%Y/%m/%d')
    balancedata.drop_duplicates(subset=['pubDate'], keep='last', inplace=True)
    stockdata = stockdata.merge(balancedata, left_on='date', right_on='pubDate', how='left')
    stockdata.drop(columns=['pubDate'], inplace=True)

    cashflow_col_list = ['pubDate',	'CFOToNP']
    cashflowdata = pd.read_csv(r'D:/cashflowdata/%s.csv' % stock, encoding='gbk', usecols=cashflow_col_list)
    cashflowdata['pubDate'] = pd.to_datetime(cashflowdata['pubDate'], format='%Y/%m/%d')
    cashflowdata.drop_duplicates(subset=['pubDate'], keep='last', inplace=True)
    stockdata = stockdata.merge(cashflowdata, left_on='date', right_on='pubDate', how='left')
    stockdata.drop(columns=['pubDate'], inplace=True)

    stockdata.fillna(method='ffill')

    stockdata['industry'] = industry_dic[stockdata.loc[0, 'code']]

    stockdata.to_csv(r'D:/svmdata/integrated/%s.csv'%stock, encoding='gbk', index=False)

target_date_list = []
sample = pd.read_csv(r'D:/stockdata/sh.600008.csv', encoding='gbk')
sample['datetm'] = pd.to_datetime(sample['date'], format='%Y/%m/%d')
sample['date'] = sample['date'].astype(str)
current_month = sample.loc[0, 'datetm'].month
for i in range(0, sample.shape[0]-1):
    if sample.loc[i+1, 'datetm'].month != current_month:
        target_date_list.append(sample.loc[i, 'date'])
        current_month = sample.loc[i+1, 'datetm'].month

i = 1
for date in target_date_list:
    result = pd.DataFrame()
    for stock in zz500_stock_list:
        data = pd.read_csv(r'D:/svmdata/integrated/%s.csv'%stock, encoding='gbk')
        data.fillna(method='ffill', inplace=True)
        data['date'] = data['date'].astype(str)
        data.set_index('date', inplace=True)
        result = result.append(data[data.index == date])
    result.to_csv(r'D:/svmdata/bymonth/%i.csv'%i, encoding='gbk', index=False)
    i = i+1

# preprocessing
sample = pd.read_csv(r'D:/svmdata/bymonth/1.csv', encoding='gbk')
fields = list(sample.columns)
fields2 = ['code', 'close', 'industry', 'pctChg', 'isST', 'tradestatus']
for item in fields2:
    fields.remove(item)

for month in range(7, 131):
# filter according to ST and tradingstatus
    sample = pd.read_csv(r'D:/svmdata/bymonth/%i.csv'%month, encoding='gbk')
    sample = sample[sample.isST == 0]
    sample = sample[sample.tradestatus == 1]

    for field in fields:
        # outlier winsorization
        MAD = stats.median_abs_deviation(sample[field].dropna())
        median = sample[field].median()
        if not sample[field][sample[field] > median+5*MAD].empty:
            sample[field][sample[field] > median+5*MAD] = median+5*MAD
        if not sample[field][sample[field] < median-5*MAD].empty:
            sample[field][sample[field] < median-5*MAD] = median-5*MAD
        # fill na by industry mean
        sample[field] = sample.groupby('industry')[field].transform(lambda x: x.fillna(x.mean()))
        # adding two more factors

    sample['logprice'] = np.log(sample['close'])
    sample['marketvalue'] = np.log(sample['totalShare'].multiply(sample['close']))

    for field in fields:
        from statsmodels.formula.api import ols
        fit = ols(r'%s ~ C(industry) + marketvalue'%field, data=sample).fit()
        sample[field] = fit.resid

    print(month, sample.isna().any(1).sum())
    sample.dropna(inplace=True, how='any')
    sample.to_csv(r'D:/svmdata/final/%i.csv'%month, encoding='gbk', index=False)