#导入本次实验所需要用到的python库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble, tree, linear_model
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score #划分数据 交叉验证

train_data=pd.read_csv('F:\\Graduate\\Kaggle\\SalePrice\\SalePrice\\Data(Original)\\train_data\\train.csv')
test_data=pd.read_csv('F:\\Graduate\\Kaggle\\SalePrice\\SalePrice\\Data(Original)\\test_data\\test.csv')

##价格分布图----------可以暂不执行
print(train_data.shape,test_data.shape)
train_y = train_data.pop('SalePrice')
y_plot = sns.distplot(train_y)
print(train_data,train_y)

##价格对数化绘图,其目的将数据整体缩小---------可以暂不执行
#train_y_log = np.log(train_y)
#y_plot_log = sns.distplot(train_y_log)

#skewness and kurtosis--------可以暂不执行
#print("Skewness: %f" % train_y.skew())
#print("Kurtosis: %f" % train_y.kurt())

#scatter plot GrLivArea/saleprice--------可以暂不执行
#var = 'GrLivArea'
#data = pd.concat([train_y, train_data[var]], axis=1)
#data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

#scatter plot totalbsmtsf/saleprice``--------可以暂不执行
#var = 'TotalBsmtSF'
#data = pd.concat([train_y, train_data[var]], axis=1)
#data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

#box plot overallqual/saleprice`--------可以暂不执行
#var = 'OverallQual'
#data = pd.concat([train_y, train_data[var]], axis=1)
#f, ax = plt.subplots(figsize=(8, 6))
#fig = sns.boxplot(x=var, y="SalePrice", data=data)
#fig.axis(ymin=0, ymax=800000);

#OverallQual(房屋质量)与SalePrice关系`--------可以暂不执行
#var = 'YearBuilt'
#data = pd.concat([train_y, train_data[var]], axis=1)
#f, ax = plt.subplots(figsize=(16, 8))
#fig = sns.boxplot(x=var, y="SalePrice", data=data)
#fig.axis(ymin=0, ymax=800000);
#plt.xticks(rotation=90);

#Correlation matrix --------可以暂不执行
#data = pd.concat([train_data,train_y], axis=1)
##train_data=data
#corrmat = data.corr()
#f, ax = plt.subplots(figsize=(12, 9))
#sns.heatmap(corrmat, vmax=.8, square=True);

#saleprice correlation matrix--------可以暂不执行
#k = 10 #number of variables for heatmap
#cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
#print(cols)
#cm = np.corrcoef(train_data[cols].values.T)
#sns.set(font_scale=1.25)
#hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
#plt.show()

#scatterplot--------可以暂不执行
#sns.set()
#cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
#sns.pairplot(train_data[cols], size = 2.5)
#plt.show();

#put 'train_data'and ‘test_data’ together as 'fearure' to deal with,'train_data' don't contain 'SalePrice' now
features = pd.concat([train_data, test_data], keys=['train', 'test'])

#depart not impotant features
features.drop(['Utilities', 'RoofMatl', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 
               'Heating', 'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'GarageYrBlt', 
               'GarageArea', 'GarageCond', 'WoodDeckSF','OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 
               'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal'],
 axis=1, inplace=True)

#check the missing data each fearures
NAs = pd.concat([features.isnull().sum()],keys = ['Features'],axis=1)
NAs

#make up or depart the missing data each features
features['MSSubClass'] = features['MSSubClass'].astype(str)
features['MSZoning'] = features['MSZoning'].fillna(features['MSZoning'].mode()[0])
features['LotFrontage'] = features['LotFrontage'].fillna(features['LotFrontage'].mean())
features['Alley'] = features['Alley'].fillna('NOACCESS')
features['MasVnrType'] = features['MasVnrType'].fillna(features['MasVnrType'].mode()[0])
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    features[col] = features[col].fillna('NoBSMT')
features['TotalBsmtSF'] = features['TotalBsmtSF'].fillna(0)
features['Electrical'] = features['Electrical'].fillna(features['Electrical'].mode()[0])
features['KitchenAbvGr'] = features['KitchenAbvGr'].astype(str)
features['KitchenQual'] = features['KitchenQual'].fillna(features['KitchenQual'].mode()[0])
features['FireplaceQu'] = features['FireplaceQu'].fillna('NoFP')
for col in ('GarageType', 'GarageFinish', 'GarageQual'):
    features[col] = features[col].fillna('NoGRG')
features['GarageCars'] = features['GarageCars'].fillna(0.0)
features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])
features['YrSold'] = features['YrSold'].astype(str)
features['MoSold'] = features['MoSold'].astype(str)
features['TotalSF'] = features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']
features.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)


#Feature Enginner -one hot encoded 
all_dummy = pd.get_dummies(features)


#Standscale all the numerical features
features['Id'] = features['Id'].astype(str)
numerical_col = features.columns[features.dtypes!='object']
means = all_dummy.loc[:,numerical_col].mean()
std = all_dummy.loc[:,numerical_col].std()
all_dummy.loc[:,numerical_col] = (all_dummy.loc[:,numerical_col] - means)/std

train_data=features.loc['train']
test_data=features.loc['test']


alphas = [0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10]
train_x=train_data.loc[:,numerical_col]
test_x=test_data.loc[:,numerical_col]
test_scores  = []
for alpha in alphas:
    ElasticN = linear_model.ElasticNetCV(alphas=[alpha],
                                    l1_ratio=[.01, .1, .5, .9, .99],
                                    max_iter=5000)
    test_score  = cross_val_score(ElasticN, train_x, train_y, cv=5)
    test_mean=test_score.mean(axis=0)
    test_scores=np.append(test_scores,[test_mean],axis=0)
print(test_scores)

#check train_data or test_data missing data's number in each column---------------可以暂不执行
#missing_val_count_by_column = (test_data.isnull().sum())
#missing_val_count_by_column

#deal with missing data------可以暂不执行
#cols_with_missing = [col for col in train_data.columns 
                               #  if train_data[col].isnull().any()]
#train_data = train_data.drop(cols_with_missing, axis=1)
#test_data = test_data.drop(cols_with_missing, axis=1)


#Predict
ElasticN = linear_model.ElasticNetCV(alphas=[0.001],
                                    l1_ratio=[.01, .1, .5, .9, .99],
                                    max_iter=5000)
Model = ElasticN.fit(train_x, train_y)
Predict_Price_1=Model.predict(test_x)
Predict_Price_1_distplot= sns.distplot(Predict_Price_1)


Max_features = [.1,.3,.5,.7,.9,.99]
test_scores2 = []
for feature in Max_features:
    RFR = RandomForestRegressor(n_estimators = 200,max_features = feature)
    test_score  = cross_val_score(RFR, train_x, train_y, cv=5)
    test_scores2.append(test_score.mean())
test_scores2

sqrt_score2 = np.sqrt(test_scores2)
plt.scatter(Max_features,sqrt_score2)


RFR = RandomForestRegressor(n_estimators = 200,max_features = 0.3)
clf = RFR.fit(train_x,train_y)
Predict_Price_2 = clf.predict(test_x)
Predict_Price_2_distplot= sns.distplot(Predict_Price_2)

Final_SalePrice  = (Predict_Price_1+Predict_Price_2)/2
Final_SalePrice_distplot=sns.distplot(Final_SalePrice)

pd.DataFrame({'Id': test_data.Id, 'SalePrice': Final_SalePrice}).to_csv('House_Price_Prediction.csv', index =False)
print('aaaa')