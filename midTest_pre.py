import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
warnings.filterwarnings('ignore')

## 0.导入数据
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
len(train['insured_zip'].unique())
## 1.数据基本信息
print("-------查看数据信息--------")
print('fraud各类别比例：\n',(train['fraud'].value_counts() / len(train)).round(2))  # 统计目标变量比例


## 2.数据探索
y_label = train['fraud']
# 去除无用变量（编号和邮编）
train_data = train.drop(columns=['policy_id','insured_zip'])
# 转化时间格式变量（保险绑定时间和出现日期）
datetime_list = ['policy_bind_date', 'incident_date']
for val in datetime_list:
    train_data[val] = pd.to_datetime(train_data[val], format='%Y-%m-%d')
train_data.insert(0,'delta_months',(train_data['incident_date'] - train_data['policy_bind_date']).dt.days//30)
train_data = train_data.drop(columns=datetime_list)
# 将初步处理后的数据保存为 train_basic.csv
train_data.to_csv('train_basic.csv',index=False)    #index=False 去除行索引

## 3.缺失值处理

#region 众数补全（'property_damage','police_report_available'）
# 'collision_type'由数据背景转化为新取值，在数据编码过程中直接转化
train_T = train_data[train_data['fraud'] == 1]  # 519
train_F = train_data[train_data['fraud'] == 0]  # 181
train_T['property_damage'] = train_T['property_damage'].apply(lambda x: 'YES' if x == '?' else x)
train_F['property_damage'] = train_F['property_damage'].apply(lambda x: 'NO' if x == '?' else x)
train_data = pd.concat([train_T, train_F], axis=0)
train_data['police_report_available'] = train_data['police_report_available'].apply(lambda x: 'NO' if x == '?' else x)
# 把众数填充后的数据保存为 train_modeFill.csv
train_data.to_csv('train_modeFill.csv',index=False)
#endregion

# 连续特征离散化（聚类分箱）
NuC_features = ['age','customer_months','policy_annual_premium','capital-gains','capital-loss','total_claim_amount','injury_claim','property_claim','vehicle_claim','delta_months']
NuD_features = ['policy_deductable','umbrella_limit','number_of_vehicles_involved','bodily_injuries','witnesses','auto_year','incident_hour_of_the_day']


#region use Silhouette
train_afterSil = train_data.copy()
res_silhouette = []
K = range(2,15)
for i in range(len(NuC_features)):
    choose_k = []
    x = train_afterSil[NuC_features[i]].values.reshape(-1, 1)
    for k in K:
        kmodel = KMeans(n_clusters=k)
        cluster = kmodel.fit(x)
        meanSil = silhouette_score(x,cluster.labels_)
        choose_k.append(meanSil)
    res_silhouette.append(choose_k)
# 绘制
fig = plt.figure()
plt.subplots_adjust(wspace=0.4,hspace=1)
for i in range(len(NuC_features)):
    ax = fig.add_subplot(5,2,i+1)
    ax.plot(K,res_silhouette[i],color='#508CB4')
    ax.set_xlabel('k')
    ax.set_ylabel("Silhouette Score",fontsize=8)
    ax.set_title(NuC_features[i])
    #ax.grid(True)
plt.show()
# 离散化
num_sil = [12,3,2,2,2,2,3,3,2,2] # 根据碎石图选取聚类数
sil_ = {}
for i in range(len(num_sil)):
    sil_[NuC_features[i]] = num_sil[i]
NuC_features_sil = {}
for i in range(len(num_sil)):
    kmodel = KMeans(n_clusters=num_sil[i])
    NuC_features_sil[NuC_features[i]+'_dis'] = kmodel.fit_predict(train_afterSil[NuC_features[i]].values.reshape(-1,1))
    #train_data[NuC_features[i]+'_dis'] = kmodel.fit_predict(train_data[NuC_features[i]].values.reshape(-1,1))
    train_afterSil.insert(0,NuC_features[i]+'_dis',kmodel.fit_predict(train_afterSil[NuC_features[i]].values.reshape(-1,1)))
train_afterSil.to_csv("train_afterSil.csv",index=False)

#endregion




#region use Average Dispersion
train_afterAD = train_data.copy()
res_AD = []
K  = range(2,15)
for i in range(len(NuC_features)):
    chose_k = []
    x = train_afterAD[NuC_features[i]].values.reshape(-1,1)
    for k in K:
        kmodel = KMeans(n_clusters=k)
        kmodel.fit(x)
        chose_k.append(sum(np.min(cdist(x,kmodel.cluster_centers_,'euclidean'),axis=1))/x.shape[0])
    res_AD.append(chose_k)

# 绘制碎石图
fig = plt.figure()
plt.subplots_adjust(wspace=0.4,hspace=1)
for i in range(len(NuC_features)):
    ax = fig.add_subplot(5,2,i+1)
    ax.plot(K,res_AD[i],color='#508CB4')
    ax.set_xlabel('k')
    ax.set_ylabel("Average Dispersion",fontsize=8)
    ax.set_title(NuC_features[i])
    #ax.grid(True)
plt.show()

# 聚类离散化
num_AD = [6,8,8,2,2,4,3,3,4,4] # 根据碎石图选取聚类数
kmean_relation = {}
for i in range(len(num_AD)):
    kmean_relation[NuC_features[i]] = num_AD[i]
NuC_features_AD = {}
for i in range(len(num_AD)):
    kmodel = KMeans(n_clusters=num_AD[i])
    NuC_features_AD[NuC_features[i]+'_dis'] = kmodel.fit_predict(train_afterAD[NuC_features[i]].values.reshape(-1,1))
    #train_data[NuC_features[i]+'_dis'] = kmodel.fit_predict(train_data[NuC_features[i]].values.reshape(-1,1))
    train_afterAD.insert(0,NuC_features[i]+'_dis',kmodel.fit_predict(train_afterAD[NuC_features[i]].values.reshape(-1,1)))
train_afterAD.to_csv("train_afterAD.csv",index=False)

#endregion