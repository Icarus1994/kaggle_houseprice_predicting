from data_exploring import *
import pandas as pd

#定性特征转化为定量特征
Functional_mapping = {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, "Min2" : 6, "Min1" : 7, "Typ" : 8}
alldata['Functional'] = alldata['Functional'].map(Functional_mapping)
test['Functional'] = test['Functional'].map(Functional_mapping)


alldata["SimplFunctional"] = alldata.Functional.replace(
    {1 : 1, 2 : 1,
     3 : 2, 4 : 2,
     5 : 3, 6 : 3, 7 : 3,
     8 : 4 })
test["SimplFunctional"] = test.Functional.replace(
    {1 : 1, 2 : 1,
     3 : 2, 4 : 2,
     5 : 3, 6 : 3, 7 : 3,
     8 : 4 })

# alldata["OverallQual-s2"] = alldata["OverallQual"] ** 2
# alldata["OverallQual-s3"] = alldata["OverallQual"] ** 3
# alldata["OverallQual-Sq"] = np.sqrt(alldata["OverallQual"])
#
# alldata["OverallGrade"] = alldata["OverallQual"] * alldata["OverallCond"]

#convert categorical variable into dummy
# print(alldata.head(3))
# print(test.head(5))

alldata = pd.get_dummies(alldata)



