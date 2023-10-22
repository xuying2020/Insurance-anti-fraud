import matplotlib.pyplot as plt  # 导入绘图库
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv("train.csv")

'''堆积柱状图'''
Ca_feature = list(train.select_dtypes(include=['object']).columns)  # 分类数据
Ca_feature.remove('policy_bind_date')
Ca_feature.remove('insured_occupation')
Ca_feature.remove('incident_date')
Ca_feature.remove('insured_hobbies')
Ca_feature.remove('auto_make')
Ca_feature.remove('auto_model')
Ca_feature.remove('incident_city')

plt.figure(figsize=(15, 10))
rgbcolor  = [(80/255,140/255,180/255),(215/255,228/255,239/255)]
i = 1
for col in Ca_feature:
    ax = plt.subplot(3, 4, i)
    df_group = pd.crosstab(train[col], train['fraud'])
    labels = df_group.index.values
    fraud0 = []
    fraud1 = []
    for j in range(len(labels)):
        fraud0.append(df_group.iat[j, 0])
        fraud1.append(df_group.iat[j, 1])
    width = 0.5
    ax.bar(labels, fraud0, width, label='fraud0',color=rgbcolor[0])
    ax.bar(labels, fraud1, width, bottom=fraud0, label='fraud1',color=rgbcolor[1])
    ax.set_xticklabels(labels, rotation=20)
    ax.legend()
    ax.set_title(col)
    i = i + 1
# plt.show()
plt.tight_layout()
plt.savefig("Ca_feature.png", bbox_inches='tight')

# 分类超过10个的单独作图 insured_occupation,insured_hobbies,auto_make,auto_model,incident_city
plt.figure(figsize=(15, 10))
ax = plt.subplot()
rgbcolor  = [(80/255,140/255,180/255),(215/255,228/255,239/255)]
df_group = pd.crosstab(train['incident_city'], train['fraud'])
labels = df_group.index.values
fraud0 = []
fraud1 = []
for j in range(len(labels)):
    fraud0.append(df_group.iat[j, 0])
    fraud1.append(df_group.iat[j, 1])
width = 0.5
ax.bar(labels, fraud0, width, label='fraud0',color=rgbcolor[0])
ax.bar(labels, fraud1, width, bottom=fraud0, label='fraud1',color=rgbcolor[1])
ax.set_xticklabels(labels, rotation=20)
ax.legend()
ax.set_title('incident_city')
# plt.show()
# plt.tight_layout()
plt.savefig("incident_city.png", bbox_inches='tight')

'''概率密度函数图，查看变量的分布'''
# 将变量分类
numercial_feature = list(train.select_dtypes(exclude=['object']).columns)  # 数值变量
Ca_feature = list(train.select_dtypes(include=['object']).columns)
# 连续型变量
serial_feature = []
# 离散型变量
discrete_feature = []
# 单值变量
unique_feature = []
for feature in numercial_feature:
    temp = train[feature].nunique()  # 返回数据去重后的个数
    if temp == 1:
        unique_feature.append(feature)
    elif 1 < temp <= 10:
        discrete_feature.append(feature)
    else:
        serial_feature.append(feature)
# print("-------将数据分类--------")
# print('serial_feature:', serial_feature)
# print('discrete_feature:', discrete_feature)
# print('unique_feature:', unique_feature)

serial_feature.remove('policy_id')
serial_feature.remove('umbrella_limit')
serial_feature.remove('insured_zip')
serial_feature.remove('policy_annual_premium')
serial_feature.remove('auto_year')

serial_df = pd.melt(train, value_vars=serial_feature)  # 将连续型变量融合在一个dataframe中
f = sns.FacetGrid(serial_df, col='variable', col_wrap=3, sharex=False, sharey=False)  # 生成画布，最多三列，不共享x、y轴
f.map(sns.distplot, "value")
#plt.show()
plt.savefig("serial_feature.png", bbox_inches='tight')

'''箱线图'''
Nu_feature = list(train.select_dtypes(exclude=['object']).columns)  # 数值变量
Ca_feature = list(train.select_dtypes(include=['object']).columns)
Nu_feature.remove('policy_id')
Nu_feature.remove('policy_deductable')
Nu_feature.remove('umbrella_limit')
Nu_feature.remove('insured_zip')
Nu_feature.remove('bodily_injuries')
Nu_feature.remove('fraud')
Nu_feature.remove('witnesses')
Nu_feature.remove('number_of_vehicles_involved')
print(Nu_feature)
# 绘制箱线图
plt.figure(figsize=(30, 30))  # 箱线图查看数值型变量异常值
i = 1
for col in Nu_feature:
    ax = plt.subplot(4, 4, i)
    ax = sns.boxplot(x="fraud", y=col, data=train[[col, 'fraud']],boxprops={'color':'#508CB4','facecolor':'#508CB4'})
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    i += 1
#plt.show()
plt.savefig("boxplot.png", bbox_inches='tight')

'''heatmap图'''
numercial_feature = list(train.select_dtypes(exclude=['object']).columns)  # 数值变量
numercial_feature.remove('policy_id')
numercial_feature.remove('insured_zip')
numercial_feature.remove('fraud')

cor = train[numercial_feature].corr()
sns.set_theme(style="white")
plt.figure(figsize=(16, 10))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(cor, cmap=cmap, annot=True, linewidth=0.2,
            cbar_kws={"shrink": .5}, linecolor="white", fmt=".1g")
plt.xticks(rotation=75)
plt.savefig("heatmap.png", bbox_inches='tight')
# plt.show()

'''sex 饼图'''
a = train.insured_sex.value_counts()
plt.pie(a, labels=a.index, autopct='%.1f%%',
        wedgeprops=dict(width=0.3, edgecolor='w'),colors=['#508CB4','#d7e4ef'])
# plt.show()
plt.savefig("insured_sex.png", bbox_inches='tight')