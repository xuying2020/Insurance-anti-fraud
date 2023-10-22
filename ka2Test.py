# property_damage：是否有财产损失缺失值，可能是数据未记录
# police_report_available：是否有警察记录的报告缺失值，可能是数据未记录
import pandas as pd
from scipy.stats import chi2

fraud = pd.read_csv("train.csv")['fraud']
df = pd.read_csv("train_new.csv")

CaFeatures = ['policy_state', 'policy_csl', 'insured_sex', 'insured_education_level', 'insured_occupation',
              'insured_hobbies', 'insured_relationship', 'incident_type', 'collision_type', 'incident_severity',
              'authorities_contacted', 'incident_state', 'incident_city', 'property_damage', 'police_report_available',
              'auto_make', 'auto_model']
cat_1 = df.policy_state
cat_2 = df.policy_csl
cat_3 = df.insured_sex
cat_4 = df.insured_education_level
cat_5 = df.insured_occupation
cat_6 = df.insured_hobbies
cat_7 = df.insured_relationship
cat_8 = df.incident_type
cat_9 = df.collision_type
cat_10 = df.authorities_contacted
cat_11 = df.incident_state
cat_12 = df.incident_city
cat_13 = df.authorities_contacted
cat_14 = df.property_damage
cat_15 = df.police_report_available
cat_16 = df.auto_make
cat_17 = df.auto_model

df_1 = pd.crosstab(cat_1, fraud, margins=True)
df_2 = pd.crosstab(cat_2, fraud, margins=True)
df_3 = pd.crosstab(cat_3, fraud, margins=True)
df_4 = pd.crosstab(cat_4, fraud, margins=True)
df_5 = pd.crosstab(cat_5, fraud, margins=True)
df_6 = pd.crosstab(cat_6, fraud, margins=True)
df_7 = pd.crosstab(cat_7, fraud, margins=True)
df_8 = pd.crosstab(cat_8, fraud, margins=True)
df_9 = pd.crosstab(cat_9, fraud, margins=True)
df_10 = pd.crosstab(cat_10, fraud, margins=True)
df_11 = pd.crosstab(cat_11, fraud, margins=True)
df_12 = pd.crosstab(cat_12, fraud, margins=True)
df_13 = pd.crosstab(cat_13, fraud, margins=True)
df_14 = pd.crosstab(cat_14, fraud, margins=True)
df_15 = pd.crosstab(cat_15, fraud, margins=True)
df_16 = pd.crosstab(cat_16, fraud, margins=True)
df_17 = pd.crosstab(cat_17, fraud, margins=True)


def compute_S(my_df):
    S = []
    for i in range(2):
        for j in range(2):
            E = my_df.iat[i, j]
            F = my_df.iat[i, 2] * my_df.iat[2, j] / my_df.iat[2, 2]
            S.append((E - F) ** 2 / F)
    return sum(S)


res1 = compute_S(df_1)
res2 = compute_S(df_2)
res3 = compute_S(df_3)
res4 = compute_S(df_4)
res5 = compute_S(df_5)
res6 = compute_S(df_6)
res7 = compute_S(df_7)
res8 = compute_S(df_8)
res9 = compute_S(df_9)
res10 = compute_S(df_10)
res11 = compute_S(df_11)
res12 = compute_S(df_12)
res13 = compute_S(df_13)
res14 = compute_S(df_14)
res15 = compute_S(df_15)
res16 = compute_S(df_16)
res17 = compute_S(df_17)
print('policy_state检验的p值', chi2.sf(res1, 1))
# policy_state 检验的p值
# 认为不相关，删除
# 0.14959885688202387
print('policy_csl检验的p值', chi2.sf(res2, 1))
# policy_csl 检验的p值
# 认为不相关，剔除
# 0.13630883165237975
print('insured_sex检验的p值', chi2.sf(res3, 1))
# insured_sex 检验的p值
# 认为不相关，剔除
# 0.5254974340467571
print('insured_education_level检验的p值', chi2.sf(res4, 1))
# insured_education_level 检验的p值
# 认为不相关，剔除
# 0.23590886579384218
print('insured_occupation检验的p值', chi2.sf(res5, 1))
# insured_occupation 检验的p值
# 认为不相关，剔除
# 0.22222348059367586
print('insured_hobbies检验的p值', chi2.sf(res6, 1))
# insured_hobbies 检验的p值
# 认为不相关，剔除
# 0.11888648789916778
print('insured_relationship检验的p值', chi2.sf(res7, 1))
# insured_relationship 检验的p值
# 认为相关，保留
# 0.010678475583782737
print('incident_type检验的p值', chi2.sf(res8, 1))
# incident_type 检验的p值
# 认为相关，保留
# 0.006448133359998604
print('collision_type检验的p值', chi2.sf(res9, 1))
# collision_type 检验的p值
# 认为相关，保留
# 5.608043366848733e-09
print('incident_severity检验的p值', chi2.sf(res10, 1))
# incident_severity 检验的p值
# 认为相关，保留
# 1.517777088623107e-56
print('authorities_contacted检验的p值', chi2.sf(res11, 1))
# authorities_contacted 检验的p值
# 认为相关，保留
# 5.667726124540857e-09
print('incident_state检验的p值', chi2.sf(res12, 1))
# incident_state 检验的p值
# 认为不相关，剔除
# 0.5980789545686709
print('incident_city检验的p值', chi2.sf(res13, 1))
# incident_city 检验的p值
# 认为相关，保留
# 1.517777088623107e-56
print('property_damage检验的p值', chi2.sf(res14, 1))
# property_damage 检验的p值
# 认为相关，保留
# 0.02154569984730226
print('police_report_available检验的p值', chi2.sf(res15, 1))
# police_report_available 检验的p值
# 认为不相关，剔除
# 0.26630484425303813
print('auto_make检验的p值', chi2.sf(res16, 1))
# auto_make 检验的p值
# 认为相关，保留
# 0.02665867136635486
print('auto_model检验的p值', chi2.sf(res17, 1))
# auto_model 检验的p值
# 认为不相关，剔除
# 0.13126205678370836
