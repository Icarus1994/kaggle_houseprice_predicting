import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import *

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
alldata = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'], test.loc[:,'MSSubClass':'SaleCondition']), ignore_index=True)
#返回dataframe的维度
#print(alldata.shape)
# list1 = train.columns.values.tolist()
# list2 = test.columns.values.tolist()
# for element in list1:
#     if element not in list2:
#         print(element)

'''
缺失值处理
explore = train.describe(include = 'all').T
#测试值哪些列有缺失值,null的值表示某列缺失的个数
explore['null'] = len(train) - explore['count']
explore.insert(0,'dtype',train.dtypes)
explore.T.to_csv('explore1.csv')

explore = alldata.describe(include = 'all').T
explore['null'] = len(alldata) - explore['count']
explore.insert(0,'dtype',alldata.dtypes)
explore.T.to_csv('explore2.csv')
'''


# 相关图
#求相关性矩阵
corrmat = train.corr()
#figsize：整型元组，可选参数 ，默认：None.每英寸的宽度和高度
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
#pycharm加上这一句作图才能显示
#plt.show()

#查看影响最终价格的十个变量
k = 10
plt.figure(figsize=(12,9))
#cols的数据类型是什么
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
#print(cols.values)
#通过cols选中train中多列组成的dataframe
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                 annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
#显示之前所有的图
#plt.show()



#处理缺失值
total = alldata.isnull().sum().sort_values(ascending=False)
percent = (alldata.isnull().sum()/alldata.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
#print(missing_data.head(30))


#丢弃缺失值
#dataframe.index是index
#train.drop(labels=labels1,axis=1)
alldata = alldata.drop((missing_data[missing_data['Total'] > 4]).index,1)
alldata = alldata.fillna(method = "bfill", axis=0)
null_cnt = alldata.isnull().sum().sort_values(ascending=False)
#print(null_cnt)


#处理outliers
#bivariate analysis saleprice/grlivarea
#在和房价相关性排前20的变量中对每个变量与房价画散点图，看是否有异常值
# var = 'GrLivArea'
# data = pd.concat([train['SalePrice'], train[var]], axis=1)
# data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
# #deleting points
# train.sort_values(by = 'GrLivArea', ascending = False)[:2]
# train = train.drop(train[alldata['Id'] == 1299].index)
# train = train.drop(train[alldata['Id'] == 524].index)
#
# alldata = alldata[alldata['GarageArea']<1200]


#转化为正态分布的变量
#histogram and normal probability plot
#弄清楚sns.distplot是受了什么影响,缺少了正太分布图
sns.distplot(train['SalePrice'], fit = norm)
fig = plt.figure()
res = probplot(train['SalePrice'], plot=plt)
#plt.show()

#applying log transformation
#scatter plot
plt.scatter(train['GrLivArea'], train['SalePrice'])
#data transformation
alldata['GrLivArea'] = np.log(alldata['GrLivArea'])





