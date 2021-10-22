#!/usr/bin/env python
# coding: utf-8

# In[156]:


# Genetic algorithm to find microbiome discovery in IRFL dataset


# In[157]:


get_ipython().magic('matplotlib inline')


# Package
import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
from sklearn.metrics import accuracy_score


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from deap import creator, base, tools, algorithms
import sys

# random sampling
from random import sample

# combination
from itertools import combinations

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

# ROC analysis
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from scipy import interp

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.stats import ttest_ind
import xgboost as xgb
from sklearn.metrics import f1_score


import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


# hyperparameter for genetic algorithm
random.seed(42)
_functional_relevance_lookup_dict = {}
_population = 300
_generation = 100
_penalty_weight = 10
_num_one = 9
_cxpb = 0.8
_mutpb = 0.003
_n_fold = 3


def average(l):
    """
    Returns the average between list elements
    """
    return (sum(l)/float(len(l)))


def evaluate(individual, X, y):

    if(individual.count(0) != len(individual)):
        cols = [i for i in range(len(individual)) if individual[i] == 1]

        X = X.iloc[:,cols]
                # RPM
        for x in range(0, len(X)):
            X.iloc[x] = 1000000 * X.iloc[x] / X.sum(axis=1)[x]
        
        X = X.dropna()
        y = y.loc[X.index]

        try:
            if len(y.drop_duplicates()) == 2:
                # define the model
                model = RandomForestClassifier()
                # evaluate the model
                cv = RepeatedStratifiedKFold(n_splits=_n_fold, n_repeats=1, random_state=1)
                y = label_encoder.fit_transform(y)
                n_scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=3, error_score='raise')
                avg_roc_score = mean(n_scores)
                penalty_function = _penalty_weight * abs(individual.count(1) - 10)

                return avg_roc_score - penalty_function,
            else:
                return -100000,
        except ValueError:
            return -100000,
    else:
        # return 10000, 10000 #, 10000
        return -100000,



# data loading
otu_table = pd.read_csv(workdir + "//toydata.csv",index_col = ["patient"])

analysis_table = otu_table
y = analysis_table["class_1"].copy()
analysis_table = analysis_table.drop(["class_1"], axis=1)

X = analysis_table

print("IRFL in training set : %s" % len(y[y == "IRFL"]))
print("IRNF in training set : %s" % len(y[y == "IRNF"]))


# convert class into number
for x in y.index:
    if y.loc[x] == "IRFL":
        y.loc[x] = 1
    else:
        y.loc[x] = 0


# add fitness function to genetic algorithm platform
creator.create("MyFitnessMulti", base.Fitness, weights=(1.0,)) # Note here <- I used only two weights!  (at first, I tried weights=(-1.0 , -1.0, 1.0)) but it crashes. With deap, you cannot do such a thing.
creator.create("Individual", list, fitness=creator.MyFitnessMulti)

toolbox = base.Toolbox()

# Attribute generator
def random_individual_three():
    zero_list = [0 for i in range(len(X.columns))]
    index_for_one = np.random.choice(len(X.columns), _num_one, replace=True)
#     index_for_one = random.sample(range(len(X.columns)),_num_one) # 중복을 허용하지 않는 index random sampling
    # index_for_one = list(combinations(range(len(X.columns)),_num_one)) # combination 생성
    for i in index_for_one:
        zero_list[i]= 1 
    
#     print(zero_list.count(1))
    return zero_list
    

# Structure initializers
toolbox.register("rand_ind",random_individual_three)
toolbox.register("individual", tools.initIterate, creator.Individual
                 ,toolbox.rand_ind) #, n=30) #
# toolbox.register("individual", tools.initIterate, creator.Individual, random.sample(range(len(X.columns)),_num_one))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Registration of crossover & mutation functions
#toolbox.register("mate", tools.cxUniform, indpb=0.8)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=_mutpb)

# Registration of selection function
toolbox.register("select", tools.selBest)


toolbox.register("evaluate", evaluate, X=X, y=y)
logbook = tools.Logbook()



# add function for selecting the best individual
pop = toolbox.population(n=_population)
hof = tools.HallOfFame(1) # a ParetoFront may be used to retrieve the best non dominated individuals of the evolution

# add function for statistics
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)

# initialization of population
population = []

# run genetic algorithm
for i in range(0,_generation + 1):
    print ("Generation: %d" % i)
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=_cxpb, mutpb=_mutpb, ngen=1, stats=stats, halloffame=hof, verbose=True)
    population.append(pop)


# Best individual selection
_individual = tools.selBest(hof, 1)[0]
# _individual = tools.selNSGA2(hof, 1)[0]
_individual_features = [X.columns[i] for i in range(len(_individual)) if _individual[i] == 1]
print("The best individual is :" + str(_individual))
print("The best features are: \t" + str(_individual_features))

# Fitness score
print('Fitness score: \t' + str(_individual.fitness))

print('Number of Features in Subset: \t' + str(_individual.count(1)))

print('Individual: \t' + str(_individual))

print('Feature Subset\t: ' + str(_individual_features))

for x in _individual_features:
    print ("'{0}', ".format(x), end='')


# model evaluation with selected feature using GA in training set

otu_table = pd.read_csv(workdir + "//toydata.csv",index_col = ["patient"])
train_otu = otu_table
y = train_otu["class_1"].copy()
X_feat = train_otu.drop(["class_1"], axis=1)
X_feat = X_feat.loc[:,_individual_features]

for x in range(0, len(X_feat)):
    X_feat.iloc[x] = 1000000 * X_feat.iloc[x] / X_feat.sum(axis=1)[x]


# convert class into number
for x in y.index:
    if y.loc[x] == "IRFL":
        y.loc[x] = 1
    else:
        y.loc[x] = 0

# drop na

X_feat = X_feat.dropna()
y = y.loc[X_feat.index]

# define the model
model = RandomForestClassifier()
y = label_encoder.fit_transform(y)
# Run classifier with ten fold cross-validation and plot ROC curves
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=89)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig3, ax3 = plt.subplots(figsize=(7, 7))    
fig4, ax4 = plt.subplots(figsize=(7, 7))    

i = 0
for train, test in cv.split(X_feat, y):
    probas_ = model.fit(X_feat.values[train],y[train]).predict_proba(X_feat.values[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    y_pred = model.predict(X_feat.values[test])
    print("%d fold accuracy: %f " %(i, accuracy_score(y[test],y_pred)))
    print("%d fold F1-score: %f " %(i, f1_score(y[test],y_pred)))
#     ax4.plot(fpr, tpr, lw=1, alpha=0.3,
#              label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    
#     i += 1

    ax3.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
y_tot_pred = model.predict(X_feat)


print("Total training accuracy: %f" % accuracy_score(y, y_tot_pred))
print("F1-score (training set) : %0.4f" % f1_score(y,y_tot_pred))

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)

ax3.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $//pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax3.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$//pm$ 1 std. dev.')

ax3.set_xlim([-0.05, 1.05])
ax3.set_ylim([-0.05, 1.05])
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic example')
ax3.legend(loc="lower right")
ax3.grid(False)
fig3.savefig("toydata_auroc_cv.svg",dpi=300)


ax4.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)

ax4.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $//pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax4.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$//pm$ 1 std. dev.')

ax4.set_xlim([-0.05, 1.05])
ax4.set_ylim([-0.05, 1.05])
ax4.set_xlabel('False Positive Rate')
ax4.set_ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic example')
ax4.legend(loc="lower right")
ax4.grid(False)
fig4.savefig("toydata_auroc_cv.svg",dpi=300)


print(_individual_features)


importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1][:len(_individual_features)]

# Print the feature ranking
print("Feature ranking:")

#for f in range(x_train.shape[1]):
for f in range(len(_individual_features)):
    print("%d. feature %d (%f) - %s" % (f + 1, indices[f], importances[indices[f]], _individual_features[f]))

# Plot the feature importances of the forest

    
fig1, ax1 = plt.subplots(figsize=(5, 5))

ax1.bar(range(len(_individual_features)), importances[indices],
        color="hotpink", yerr=std[indices], align="center")
fig1.show()
