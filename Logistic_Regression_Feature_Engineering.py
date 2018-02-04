import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import cross_val_score
import numpy as np
import pandas as pd

# Feature Engineering

train = pd.read_json("train.json")

train['inc_angle'] = train['inc_angle'].replace('na', 0.).astype(np.float32)

train.head()

def band_mean(band):
    band = np.array(band)
    return band.mean()


train['band_1_mean'] = train['band_1'].apply(band_mean)
train['band_2_mean'] = train['band_2'].apply(band_mean)


def band_median(band):
    band = np.array(band)
    return np.median(band)


train['band_1_median'] = train['band_1'].apply(band_median)
train['band_2_median'] = train['band_2'].apply(band_median)


def band_max(band):
    band = np.array(band)
    return band.max()


train['band_1_max'] = train['band_1'].apply(band_max)
train['band_2_max'] = train['band_2'].apply(band_max)


def band_min(band):
    band = np.array(band)
    return band.min()


train['band_1_min'] = train['band_1'].apply(band_min)
train['band_2_min'] = train['band_2'].apply(band_min)


def band_variance(band):
    band = np.array(band)
    return band.var()


train['band_1_variance'] = train['band_1'].apply(band_variance)
train['band_2_variance'] = train['band_2'].apply(band_variance)


def band_size(band):
    band = np.array(band)
    return np.sum(band > np.mean(band) + np.std(band)) / float(len(band))


train['band_1_size'] = train['band_1'].apply(band_size)
train['band_2_size'] = train['band_2'].apply(band_size)

X = train[['inc_angle',
           'band_1_mean', 'band_2_mean',
           'band_1_median', 'band_2_median',
           'band_1_min', 'band_2_min',
           'band_1_max', 'band_2_max',
           'band_1_variance', 'band_2_variance',
           'band_1_size', 'band_2_size']]
y = train['is_iceberg'].values

# Instantiate logistic regression model

scoreList = []
cList = []

for i in range(1, 1000):
    c = 1 / i
    lr = LogisticRegression(C=c)
    scores = cross_val_score(lr, X, y, cv=5, scoring='neg_log_loss')
    scoreList.append(scores.mean() * -1)
    cList.append(c)
for i in range(1, 1000):
    c = 1 + 1 / i
    lr = LogisticRegression(C=c)
    scores = cross_val_score(lr, X, y, cv=5, scoring='neg_log_loss')
    scoreList.append(scores.mean() * -1)
    cList.append(c)

cs = []
ci = 10**-5
for i in range(20):
    ci = ci*5
    cs.append(ci)

for c in cs:
    lr = LogisticRegression(C=c)
    scores = cross_val_score(lr, X, y, cv=5, scoring='neg_log_loss')
    scoreList.append(scores.mean() * -1)
    cList.append(c)

print(np.min(scoreList))
index = scoreList.index(np.min(scoreList))
print(cList[index])

print(scores)
print(scores.mean())
print(scores * -1)
print(scores.mean() * -1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

train_dataset = lgb.Dataset(X_train, y_train)
test_dataset = lgb.Dataset(X_test, y_test)

params = {
       'objective': 'binary',
       'metric': 'binary_logloss',
       'boosting': 'gbdt',
       'learning_rate': 0.1,
       'num_rounds': 200,
       'early_stopping_rounds': 10,
}
model = lgb.train(params, train_dataset, valid_sets=test_dataset, verbose_eval=5)

# Maximum Pixel Value Distributions

import seaborn as sns
sns.set(color_codes=True)


x1 = train['band_2_max'][train['is_iceberg'] == 1]
x2 = train['band_2_max'][train['is_iceberg'] == 0]
sns.distplot(x1, label='Iceberg')
sns.distplot(x2, label='Ship')

plt.xlabel('Maximum Pixel Intensity, Band 2 (dB)')
plt.ylabel('Density')
plt.title('Feature Engineering: Maximum Pixel Intensity, Band 2')
plt.legend()

plt.show()