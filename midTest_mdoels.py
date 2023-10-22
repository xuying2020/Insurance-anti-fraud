import pandas as pd
import warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import roc_auc_score,matthews_corrcoef
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

dataOri = pd.read_csv('train.csv')    #原始数据
dataBasic = pd.read_csv('train_basic.csv')  #基本数据
dataModefill = pd.read_csv('train_modeFill.csv')   #众数填充后数据
dataAD = pd.read_csv('train_afterAD.csv')   #AD离散化后的数据
dataSil = pd.read_csv('train_afterSil.csv') #Sil离散化后的数据
Nu_feature = list(dataBasic.select_dtypes(exclude=['object']).columns)  # 数值变量
Ca_feature = list(dataBasic.select_dtypes(include=['object']).columns)  # 分类变量
NuC_features = ['age','customer_months','policy_annual_premium','capital-gains','capital-loss','total_claim_amount','injury_claim','property_claim','vehicle_claim','delta_months']
NuD_features = ['policy_deductable','umbrella_limit','number_of_vehicles_involved','bodily_injuries','witnesses','auto_year','incident_hour_of_the_day']

#region using basic feature
encoder = LabelEncoder()
dataBasic_copy = dataBasic.copy()
dataBasic_copy[Ca_feature] = dataBasic_copy[Ca_feature].astype('str')
for val in Ca_feature:
    encoder.fit(dataBasic_copy[val])
    dataBasic_copy[val] = encoder.transform(dataBasic_copy[val])
Y = dataBasic_copy['fraud']
dataBasic_copy.to_csv('train_basic_encoder.csv',index=False)
Xbasic = dataBasic_copy.drop(columns='fraud')
Xbasic_train,Xbasic_test,Y_train,Y_test = train_test_split(Xbasic,Y,test_size=0.3)  #数据划分
# RandomForest
trees_num = 200
forestBasic = RFC(criterion='entropy',n_estimators=trees_num)
forestBasic.fit(Xbasic_train,Y_train)
YsimBasic = forestBasic.predict(Xbasic_test)
error_count_Basic = np.sum(np.abs(Y_test-YsimBasic))
accuracyBasic = 1-error_count_Basic/float(Y_test.shape[0])
mccBasic = matthews_corrcoef(Y_test,YsimBasic)
print('**********************************************************')
print('The results by using all original features:')
print('**********************************************************')
print('Number of predicted wrong sample:',error_count_Basic)
print('Accuracy is:',accuracyBasic)
print('AUC:',roc_auc_score(Y_test,YsimBasic))
print('matthew_corrcoef:',mccBasic)
#endregion

#region using basic features with modefill
dataModefill_copy = dataModefill.copy()
dataModefill_copy[Ca_feature] = dataModefill_copy[Ca_feature].astype('str')
for val in Ca_feature:
    encoder.fit(dataModefill_copy[val])
    dataModefill_copy[val] = encoder.transform(dataModefill_copy[val])
dataModefill_copy.to_csv('train_modeFill_encoder.csv',index=False)
Xmode = dataModefill_copy.drop(columns='fraud')
Xmode_train,Xmode_test,Y_train,Y_test = train_test_split(Xmode,Y,test_size=0.3)
# RandomForest
trees_num = 200
forestMode = RFC(criterion='entropy',n_estimators=trees_num)
forestMode.fit(Xmode_train,Y_train)
YsimMode = forestMode.predict(Xmode_test)
error_count_Mode = np.sum(np.abs(Y_test-YsimMode))
accuracyMode = 1-error_count_Mode/float(Y_test.shape[0])
mccMode = matthews_corrcoef(Y_test,YsimMode)
print('**********************************************************')
print('The results by using all original features with mode:')
print('**********************************************************')
print('Number of predicted wrong sample:',error_count_Mode)
print('Accuracy is:',accuracyMode)
print('AUC:',roc_auc_score(Y_test,YsimMode))
print('matthew_corrcoef:',mccMode)
#endregion

#region using basic features with AD-discretize
dataAD_copy = dataAD.copy()
NuCdis = [val+'_dis' for val in NuC_features]
dataAD_copy = pd.concat([dataAD_copy[NuCdis],dataAD_copy[NuD_features],dataAD_copy[Ca_feature]],axis=1)
for val in Ca_feature:
    encoder.fit(dataAD_copy[val])
    dataAD_copy[val] = encoder.transform(dataAD_copy[val])
dataAD_copy.to_csv('train_afterAD_encoder.csv',index=False)
XAD_train,XAD_test,Y_train,Y_test = train_test_split(dataAD_copy,Y,test_size=0.3)
# RandomForest
trees_num = 200
forestAD = RFC(criterion='entropy',n_estimators=trees_num)
forestAD.fit(XAD_train,Y_train)
YsimAD = forestAD.predict(XAD_test)
error_count_AD = np.sum(np.abs(Y_test-YsimAD))
accuracyAD = 1-error_count_AD/float(Y_test.shape[0])
mccAD = matthews_corrcoef(Y_test,YsimAD)
print('**********************************************************')
print('The results by using all features with discretize:')
print('**********************************************************')
print('Number of predicted wrong sample:',error_count_AD)
print('Accuracy is:',accuracyAD)
print('AUC:',roc_auc_score(Y_test,YsimAD))
print('matthew_corrcoef:',mccAD)
#endregion

#region using basic features with Sil-discretize
dataSil_copy = dataSil.copy()
NuCdis = [val+'_dis' for val in NuC_features]
dataSil_copy = pd.concat([dataSil_copy[NuCdis],dataSil_copy[NuD_features],dataSil_copy[Ca_feature],],axis=1)
for val in Ca_feature:
    encoder.fit(dataSil_copy[val])
    dataSil_copy[val] = encoder.transform(dataSil_copy[val])
dataSil_copy.to_csv('train_afterSil_encoder.csv',index=False)
XSil_train,XSil_test,Y_train,Y_test = train_test_split(dataSil_copy,Y,test_size=0.3)
# RandomForest
trees_num = 200
forestSil = RFC(criterion='entropy',n_estimators=trees_num)
forestSil.fit(XSil_train,Y_train)
YsimSil = forestSil.predict(XSil_test)
error_count_Sil = np.sum(np.abs(Y_test-YsimSil))
accuracySil = 1-error_count_Sil/float(Y_test.shape[0])
mccSil = matthews_corrcoef(Y_test,YsimSil)
print('**********************************************************')
print('The results by using kai-square:')
print('**********************************************************')
print('Number of predicted wrong sample:',error_count_Sil)
print('Accuracy is:',accuracySil)
print('AUC:',roc_auc_score(Y_test,YsimSil))
print('matthew_corrcoef:',mccSil)
#endregion