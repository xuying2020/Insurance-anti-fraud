import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import roc_auc_score,roc_curve,confusion_matrix,matthews_corrcoef
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold,train_test_split,RandomizedSearchCV,GridSearchCV
warnings.filterwarnings('ignore')
train = pd.read_csv('train.csv')

dataBasicEncoder = pd.read_csv('train_basic_encoder.csv')
dataModeEncoder = pd.read_csv('train_modeFill_encoder.csv')
Y = dataBasicEncoder['fraud']
X = dataBasicEncoder.drop(columns=['fraud','injury_claim','property_claim'])

#region RFECV特征选择
trees_num = 200
model = RFC(n_estimators=200)
filter_data = RFECV(estimator=model, min_features_to_select=1, step=2, cv=StratifiedKFold(10))
filter_data.fit(X,Y)
print("有效特征个数 : %d" % filter_data.n_features_)
print("全部特征等级 : %s" % list(filter_data.ranking_))
features_selected_index = [i for i,x in enumerate(list(filter_data.ranking_)) if x == 1]
features_selected = [list(X.columns)[i] for i in features_selected_index]
print("选取的特征变量：", features_selected)
X_selected = X[features_selected]
# RandomForest
X_selected_train, X_selected_test, Y_train, Y_test = train_test_split(X_selected, Y, test_size=0.3)
forest_selected = RFC(criterion='entropy', n_estimators=trees_num)
forest_selected.fit(X_selected_train, Y_train)
Y_selected_sim = forest_selected.predict(X_selected_test)
auc = roc_auc_score(Y_test, Y_selected_sim)
print('AUC:', auc)
#endregion

#region 模型调参（以变量'incident_severity','insured_hobbies'为例）
#RandomSearchCV
n_estimators = [int(x) for x in np.linspace(start=50,stop=300,num=10)]
max_fetures = ['auto','sqrt']
max_depth = [int(x) for x in np.linspace(10,100,num=10)]
max_depth.append(None)
min_samples_split = [2,5,10]
min_samples_leaf = [1,2,4]
random_grid = {'n_estimators':n_estimators,
               'max_features':max_fetures,
               'max_depth':max_depth,
               'min_samples_split':min_samples_split,
               'min_samples_leaf':min_samples_leaf,
               'bootstrap':[True,False]}
rf = RFC()
rf_random = RandomizedSearchCV(estimator=rf,param_distributions=random_grid,n_iter=100,scoring='neg_mean_absolute_error',cv=3,verbose=2,random_state=42,n_jobs=-1)
X_selected = X[['incident_severity','insured_hobbies']]
rf_random.fit(X_selected,Y)
print(rf_random.best_params_)
#GridSearchCV
param_grid = {
    'bootstrap':[True],
    'max_depth':[20,25,30,35,40,45,50],
    'max_features': ['auto'],
    'min_samples_leaf':[2,3,4,5],
    'min_samples_split':[2,4,6,8,10,12],
    'n_estimators':[50,100,200,300,400,500]
}
grid_search = GridSearchCV(estimator=rf,param_grid=param_grid,
                           scoring='neg_mean_absolute_error',cv=3,
                           n_jobs=-1,verbose=2)
grid_search.fit(X_selected,Y)
print(grid_search.best_params_)
#endregion

#region 最终模型
X_selected_train,X_selected_test,Y_train,Y_test = train_test_split(X_selected,Y,test_size=0.3)
# trian_final = pd.concat([X_selected_train,Y_train],axis=1)
# Final_train = trian_final.to_csv('Final_train.csv',index=False)
# test_final = pd.concat([X_selected_test,Y_test],axis=1)
# Final_test = trian_final.to_csv('Final_test.csv',index=False)
# Final = pd.concat([trian_final,test_final],axis=0)
# Final.to_csv('Final.csv',index=False)
forest_selected = RFC(criterion='entropy',n_estimators=200,max_depth=35,min_samples_leaf=3,min_samples_split=8)
forest_selected.fit(X_selected_train,Y_train)
Y_selected_sim = forest_selected.predict(X_selected_test)
selected_error_count = np.sum(np.abs(Y_test-Y_selected_sim))
selected_accuracy = 1-selected_error_count/float(Y_test.shape[0])
roc_auc = roc_auc_score(Y_test,Y_selected_sim)
mcc = matthews_corrcoef(Y_test,Y_selected_sim)
print('**********************************************************')
print('The results by using selected features:')
print('**********************************************************')
print('Number of predicted wrong sample:',selected_error_count)
print('Accuracy is:',selected_accuracy)
print('AUC:',roc_auc)
print('matthew_corrcoef:',mcc)
#endregion

#region 绘制AUC
fpr,tpr,threshold = roc_curve(Y_test,Y_selected_sim)
plt.title('Validation ROC')
plt.plot(fpr,tpr,'b',label='Val AUC = %0.3f' % roc_auc,color='#508CB4')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
#endregion

#region 绘制误判样本图
votes = np.round(forest_selected.predict_proba(X_selected_test) * trees_num).astype('int')
plt.figure()
plt.ion()
x = [[0, trees_num], [0, trees_num]]
y = [[0, trees_num], [trees_num, 0]]
for i in range(len(x)):
    plt.plot(x[i], y[i], 'b')

count_special1 = 0       #记录特殊样本总数
count_special2 = 0
special_sample1 = {}     #记录特殊的样本及其投票结果
special_sample2 = {}
test_index = list(pd.DataFrame(Y_test).index)   #记录测试集中对应的行号
for i in range(len(test_index)):
    if Y_selected_sim[i] == list(Y_test)[i]:
        plt.scatter(x=votes[i, 0],y=votes[i, 1],c='red',marker='o',edgecolor='black')
    else:
        if votes[i,0] < 50:
            count_special1 += 1
            special_sample1[i] = [votes[i, 0], votes[i, 1]]
        elif votes[i,0] > 150:
            count_special2 += 1
            special_sample2[i] = [votes[i, 0], votes[i, 1]]
        plt.scatter(x=votes[i, 0],y=votes[i, 1],s=100,c='yellow',marker='s',edgecolor='green')
x = [[0.25*trees_num, 0.75*trees_num],
     [0.25*trees_num, 0.75*trees_num],
     [0.25*trees_num, 0.25*trees_num],
     [0.75*trees_num, 0.75*trees_num]]
y = [[0.25*trees_num, 0.25*trees_num],
     [0.75*trees_num, 0.75*trees_num],
     [0.25*trees_num, 0.75*trees_num],
     [0.25*trees_num, 0.75*trees_num]]
for i in range(len(x)):
    plt.plot(x[i], y[i], 'b-.')

plt.xlim(0, trees_num)
plt.ylim(0, trees_num)
plt.xticks(np.linspace(0, trees_num, 5))
plt.yticks(np.linspace(0, trees_num, 5))
plt.xlabel('Number of trees predict #0')
plt.ylabel('Number of trees predict #1')
plt.title('Predict performance of random forest')
plt.ioff()
plt.show()

#endregion

#region 特殊样本分析
print('**********************************************************')
print('特殊样本分析：')
print('**********************************************************')
print('特殊样本1总数：',count_special1,'; 特殊样本2总数：',count_special2)
print('特殊样本1及其投票：',special_sample1)
print('特殊样本2及其投票',special_sample2)
rank1 = list(special_sample1.keys())
rank2 = list(special_sample2.keys())

special_sample1_order = [test_index[i] for i in rank1]  #抽取错误样本的行号
special_sample2_order = [test_index[i] for i in rank2]

special_sample1_data = train.iloc[special_sample1_order,:]  #抽取错误样本的信息
special_sample2_data = train.iloc[special_sample2_order,:]

special_data = pd.concat([special_sample1_data,special_sample2_data],axis=0)
special_data.to_csv('special_data.csv')
print('特殊错误样本的信息：',special_data)
#endregion

