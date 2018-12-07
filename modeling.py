from feature_engineering import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


#模型训练
y = np.log(train['SalePrice'])
X = alldata[:1460]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)

lr = linear_model.LinearRegression()

model = lr.fit(X_train, y_train)

predictions = model.predict(X_test)

print('RMSE is: \n', mean_squared_error(y_test, predictions))
# print(np.exp(predictions))
#预测测试集
feats = alldata[1460:]
#feats = test_final.select_dtypes(include=[np.number]).interpolate()

# #将test中的列维度补齐为tr1一致
# list1 = X_test.columns.values.tolist()
# list2 = feats.columns.values.tolist()
# for element in list1:
#     if element not in list2:
#         feats[element] = None
# feats = feats.drop(columns = 'Condition2_PosN',axis = 1)

#建模预测
#feats = test.select_dtypes(include=[np.number]).interpolate()

predictions = model.predict(feats)
final_predictions = np.exp(predictions)

print(predictions)
print(final_predictions)
#存储结果
submission = pd.DataFrame()
submission['Id'] = test.Id
submission['SalePrice'] = final_predictions
submission.to_csv('output.csv', index=False)

