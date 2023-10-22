# 1. 调用库和模块
import numpy as np
import random
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from deap import creator, base, tools, algorithms
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


# 2. 定义适应度函数
# 这里选用的适应度函数为随机森林
def getFitness(individual, X, y):
    if individual.count(0) != len(individual):
        cols = [index for index in range(len(individual)) if individual[index] == 0]
        X_sub = np.delete(X, cols, axis=1)

        clf = RandomForestClassifier(n_estimators=200, criterion="entropy", max_depth=12)

        scores = cross_val_score(clf, X_sub,y, scoring='roc_auc', cv=3)
        Average_score = sum(scores) / len(scores)
        return (Average_score,)
    else:
        return (0,)


# 3. 定义遗传算法

def geneticAlgorithm(X, y, n_population, n_generation):
    """
    Deap global variables
    Initialize variables to use eaSimple
    """

    # create individual
    # 定义问题
    # 多目标优化：creator.create('FitnessMulti', base.Fitness, weights=(-1.0, 1.0))
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # 单目标优化，最大值问题
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # create toolbox
    # 工具箱
    toolbox = base.Toolbox()
    # 生成个体-0,1编码
    toolbox.register("attr_bool", random.randint, 0, 1)  # 0,1
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_bool, X.shape[1])
    # 生成初始种群
    toolbox.register("population", tools.initRepeat, list,
                     toolbox.individual)
    #  注册遗传算法工具-评价函数,配种选择，交叉，变异
    toolbox.register("evaluate", getFitness, X=X, y=y)  # 评价函数

    toolbox.register("mate", tools.cxOnePoint)  # 单点交叉
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)  # 位翻转变异
    toolbox.register("select", tools.selTournament, tournsize=3)  # 锦标赛选择

    # initialize parameters
    pop = toolbox.population(n=n_population)  # 生成初始种群
    hof = tools.HallOfFame(n_population * n_generation)  # 所有个体
    # 注册计算过程中需要记录的数据
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # genetic algorithm
    # 调用DEAP内置算法- 简单进化算法
    global log
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2,
                                   ngen=n_generation, stats=stats, halloffame=hof,
                                   verbose=True)
    # cxpb = 0.5      # 交叉概率
    # mutpb = 0.2     # 突变概率
    # deap.algorithms.eaSimple(population,toolbox,cxpb,mutpb,ngen[,stats,halloffame,verbose])
    # return hall of fame
    return hof


# 4.定义 最优个体

def bestIndividual(hof, X, y):
    maxAccurcy = 0.0
    for individual in hof:
        # print('tuple value = ',individual.fitness.values)
        if (individual.fitness.values[0] > maxAccurcy):
            maxAccurcy = individual.fitness.values
            _individual = individual

    _individualHeader = [list(X)[i] for i in range(len(_individual)) if _individual[i] == 1]
    return _individual.fitness.values, _individual, _individualHeader


# 主函数
if __name__ == '__main__':
    # 导入数据
    # breast = datasets.load_breast_cancer()
    # X = breast.data
    # y = breast.target
    # print(X.shape)

    X = pd.read_csv('train_new.csv')
    X.drop(columns=['injury_claim', 'property_claim'], inplace=True)
    y = pd.read_csv("train.csv")['fraud']
    print(X.shape)

    # 数据归一化
    mms = preprocessing.MinMaxScaler()
    X = mms.fit_transform(X)

    # get accuracy with all features
    individual = [1 for i in range(X.shape[1])]
    print("Auc with all features: \t" + str(getFitness(individual, X, y)) + "\n")

    # apply genetic algorithm
    n_pop = 5  # 初始种群
    n_gen = 20  # 迭代数
    hof = geneticAlgorithm(X, y, n_pop, n_gen)

    # select the best individual
    accuracy, individual, _ = bestIndividual(hof, X, y)
    print('Best Accuracy: \t' + str(accuracy))
    print('Number of Features in Subset: \t' + str(individual.count(1)))
    print('Individual: \t\t' + str(individual))

    print('\n\ncreating a new classifier with the following selected features:')

    cols = [index for index in range(len(individual)) if individual[index] == 0]
    X_selected = np.delete(X, cols, axis=1)
    selected_cols = [index for index in range(len(individual)) if individual[index] != 0]
    print(selected_cols)
    print(X_selected.shape)

    clf = RandomForestClassifier(criterion="entropy", max_depth=12)

    scores = cross_val_score(clf, X_selected, y, cv=3)
    auc = cross_val_score(clf, X_selected, y, scoring='roc_auc', cv=3)
    Average_score = sum(scores) / len(scores)
    Average_auc = sum(auc) / len(auc)
    print("Accuracy with Feature Subset: \t" + str(Average_score) + "\n")
    print("Auc with Feature Subset: \t" + str(Average_auc) + "\n")

