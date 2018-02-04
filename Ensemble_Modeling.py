import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

df_1 = pd.read_csv('submission_original.csv')
df_2 = pd.read_csv('200_ens_densenet.csv')

# Apply a < 0.05, > 0.95 clip

# If neither are sure, then choose DenseNet

clipstack = []
for i in range(len(df_1[['id']])):
    if (df_1[['is_iceberg']].values[i] < 0.05 or df_1[['is_iceberg']].values[i] > 0.95) or (df_2[['is_iceberg']].values[i] < 0.05 or df_2[['is_iceberg']].values[i] > 0.95):
        if df_2[['is_iceberg']].values[i] < 0.05 or df_2[['is_iceberg']].values[i] > 0.95:
            clipstack.append(df_2[['is_iceberg']].values[i])
        else:
            clipstack.append(df_1[['is_iceberg']].values[i])

    else:
        clipstack.append(df_2[['is_iceberg']].values[i])

# Extract values

csvals = []
for i in range(len(clipstack)):
    csvals.append(clipstack[i][0])

submission = pd.DataFrame({'id': df_1["id"], 'is_iceberg': csvals})
print(submission.head(10))

submission.to_csv('submission_clipstack1.csv', index=False)

# Explore Model Distrbutions (Plotting)

# binning
x = range(len(df_1[['id']]))[0:8424:50]
y1 = df_1[['is_iceberg']].values[0:8424:50]
y2 = df_2[['is_iceberg']].values[0:8424:50]

# Clip stacking
y3 = clipstack[0:8424:50]

plt.plot(x, y1, label='Baseline_CNN (LB:0.2377)')
plt.plot(x, y2, label='DenseNet (LB:0.1538)')
plt.plot(x, y3, label='>0.95, <0.05 ClipStack (LB:0.1845)')

plt.xlabel('Iceberg ID')
plt.ylabel('Is_Icerberg (Probability)')
plt.title('Ensemble (Stacking) Explore')

plt.legend(loc='upper left')

plt.ylim(0, 1.15)

plt.show()

# Stack with optimal logistic regression model

model = LogisticRegression(C=305175.78125)

train = pd.read_json("train.json")
train['inc_angle'] = train['inc_angle'].replace('na', 0.).astype(np.float32)

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

Xt = train[['inc_angle',
           'band_1_mean', 'band_2_mean',
           'band_1_median', 'band_2_median',
           'band_1_min', 'band_2_min',
           'band_1_max', 'band_2_max',
           'band_1_variance', 'band_2_variance',
           'band_1_size', 'band_2_size']]
yt = train['is_iceberg'].values

model.fit(Xt, yt)

test = pd.read_json("test.json")
test['inc_angle'] = test['inc_angle'].replace('na', 0.).astype(np.float32)

X = test[['inc_angle',
           'band_1_mean', 'band_2_mean',
           'band_1_median', 'band_2_median',
           'band_1_min', 'band_2_min',
           'band_1_max', 'band_2_max',
           'band_1_variance', 'band_2_variance',
           'band_1_size', 'band_2_size']]

y_pred = model.predict_proba(X)

y_pred_ice = []
for i in range(len(y_pred)):
    y_pred_ice.append(y_pred[i][1])

submission = pd.DataFrame({'id': test["id"], 'is_iceberg': y_pred_ice})
print(submission.head(10))

submission.to_csv('submission_optim_log_reg.csv', index=False)

# Explore rounding confidence

# Round for >0.95 and <0.05

df_3 = pd.read_csv('submission_1_20_2018_1.csv')

roundex = []
for i in range(len(df_3[['is_iceberg']])):
    if df_3[['is_iceberg']].values[i] < 0.05:
        roundex.append(0)
    elif df_3[['is_iceberg']].values[i] > 0.95:
        roundex.append(1)
    else:
        roundex.append(df_3[['is_iceberg']].values[i][0])

csvals = []
for i in range(len(roundex)):
    csvals.append(roundex[i])

submission = pd.DataFrame({'id': df_3["id"], 'is_iceberg': csvals})
print(submission.head(10))

submission.to_csv('submission_round_explore.csv', index=False)

# Round for >0.90 and <0.1

df_3 = pd.read_csv('submission_1_20_2018_1.csv')

roundex = []
for i in range(len(df_3[['is_iceberg']])):
    if df_3[['is_iceberg']].values[i] < 0.1:
        roundex.append(0)
    elif df_3[['is_iceberg']].values[i] > 0.9:
        roundex.append(1)
    else:
        roundex.append(df_3[['is_iceberg']].values[i][0])

csvals = []
for i in range(len(roundex)):
    csvals.append(roundex[i])

submission = pd.DataFrame({'id': df_3["id"], 'is_iceberg': csvals})
print(submission.head(10))

submission.to_csv('submission_more aggressive_round.csv', index=False)

# Round for >0.95 and <0.05 (Stack of Optimized Logistic Regression + Best Public Stack)

# Plotting distributions

df_3 = pd.read_csv('submission_1_20_2018_1.csv')
df_4 = pd.read_csv('submission_optim_log_reg.csv')

# binning
x = range(len(df_3[['id']]))[0:8424:50]
y1 = df_3[['is_iceberg']].values[0:8424:50]
y2 = df_4[['is_iceberg']].values[0:8424:50]

plt.plot(x, y1, label='Best Public Stack')
plt.plot(x, y2, label='Logistic Regression')

plt.xlabel('Iceberg ID')
plt.ylabel('Is_Icerberg (Probability)')
plt.title('Optimized L2-regularized Logistic Regression vs. Best Public Stacking')

plt.legend(loc='upper left')

plt.ylim(0, 1.15)

plt.show()

# MinMax + BestBase Stacking

cutoff_lo = 0.8
cutoff_hi = 0.2

# Choosing Best Public Stack as Best Base

out1 = pd.read_csv("submission_1_20_2018_1.csv", index_col=0)
out2 = pd.read_csv("submission_optim_log_reg.csv", index_col=0)
concat_sub = pd.concat([out1, out2], axis=1)
cols = list(map(lambda x: "is_iceberg_" + str(x), range(len(concat_sub.columns))))
concat_sub.columns = cols
concat_sub.reset_index(inplace=True)

concat_sub['is_iceberg_max'] = concat_sub.iloc[:, 1:6].max(axis=1)
concat_sub['is_iceberg_min'] = concat_sub.iloc[:, 1:6].min(axis=1)
concat_sub['is_iceberg_mean'] = concat_sub.iloc[:, 1:6].mean(axis=1)
concat_sub['is_iceberg_median'] = concat_sub.iloc[:, 1:6].median(axis=1)

concat_sub['is_iceberg_base'] = df_3['is_iceberg']
concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:,1:4] > cutoff_lo, axis=1),
                                    concat_sub['is_iceberg_max'],
                                    np.where(np.all(concat_sub.iloc[:,1:4] < cutoff_hi, axis=1),
                                             concat_sub['is_iceberg_min'],
                                             concat_sub['is_iceberg_base']))
concat_sub['is_iceberg'] = np.clip(concat_sub['is_iceberg'].values, 0.001, 0.999)
concat_sub['id'] = df_3['id']

concat_sub[['id', 'is_iceberg']].to_csv('submission_best_public_stack_LR.csv',
                                        index=False, float_format='%.6f')

# Regular Round

df_6 = pd.read_csv('submission_best_public_stack_LR.csv')

roundex = []
for i in range(len(df_6[['is_iceberg']])):
    if df_6[['is_iceberg']].values[i] < 0.05:
        roundex.append(0)
    elif df_6[['is_iceberg']].values[i] > 0.95:
        roundex.append(1)
    else:
        roundex.append(df_6[['is_iceberg']].values[i][0])

csvals = []
for i in range(len(roundex)):
    csvals.append(roundex[i])

submission = pd.DataFrame({'id': df_6["id"], 'is_iceberg': csvals})
print(submission.head(10))

submission.to_csv('submission_min_max_stack_round.csv', index=False)

# More aggressive round

df_6 = pd.read_csv('submission_best_public_stack_LR.csv')

roundex = []
for i in range(len(df_6[['is_iceberg']])):
    if df_6[['is_iceberg']].values[i] < 0.1:
        roundex.append(0)
    elif df_6[['is_iceberg']].values[i] > 0.9:
        roundex.append(1)
    else:
        roundex.append(df_6[['is_iceberg']].values[i][0])

csvals = []
for i in range(len(roundex)):
    csvals.append(roundex[i])

submission = pd.DataFrame({'id': df_6["id"], 'is_iceberg': csvals})
print(submission.head(10))

submission.to_csv('submission_min_max_stack_more aggressive_round.csv', index=False)

# Plotting Logistic Regression, Best Public Stack, MinMax + BestBase Stack

df_3 = pd.read_csv('submission_1_20_2018_1.csv')
df_4 = pd.read_csv('submission_optim_log_reg.csv')
df_5 = pd.read_csv('submission_best_public_stack_LR.csv')
df_6 = pd.read_csv('submission_min_max_stack_more aggressive_round.csv')

x = range(len(df_3[['id']]))[0:8424:50]
y1 = df_3[['is_iceberg']].values[0:8424:50]
y2 = df_4[['is_iceberg']].values[0:8424:50]
y3 = df_5[['is_iceberg']].values[0:8424:50]
y4 = df_6[['is_iceberg']].values[0:8424:50]

plt.plot(x, y1, label='Best Public Stack (Base)')
plt.plot(x, y2, label='Logistic Regression')
plt.plot(x, y3, label='MinMax + BestBase Stack')
plt.plot(x, y4, label='0.9>, 0.1< Round')

plt.xlabel('Iceberg ID')
plt.ylabel('Is_Icerberg (Probability)')
plt.title('MinMax + BestBase Stack')

plt.legend(loc='upper left')

plt.ylim(0, 1.25)

plt.show()

# Final Plot

df_1 = pd.read_csv('submission_original.csv')
df_2 = pd.read_csv('200_ens_densenet.csv')
df_3 = pd.read_csv('submission_optim_log_reg.csv')
df_4 = pd.read_csv('submission_1_20_2018_1.csv')
df_5 = pd.read_csv('submission_best_public_stack_LR.csv')


x = range(len(df_3[['id']]))[0:8424:50]
y1 = df_1[['is_iceberg']].values[0:8424:50]
y2 = df_2[['is_iceberg']].values[0:8424:50]
y3 = df_3[['is_iceberg']].values[0:8424:50]
y4 = df_4[['is_iceberg']].values[0:8424:50]
y5 = df_5[['is_iceberg']].values[0:8424:50]

plt.plot(x, y1, label='Best CNN (Base)', linewidth=0.9)
plt.plot(x, y2, label='DenseNet', linewidth=0.9)
plt.plot(x, y3, label='Logistic Regression', linewidth=0.9)
plt.plot(x, y4, label='Best Public Stack', linewidth=0.9)
plt.plot(x, y5, label='MinMax + BestBase Stack', linewidth=2.0)

plt.xlabel('Image ID')
plt.ylabel('Is_Icerberg (Probability)')
plt.title('Ensemble Modeling')

plt.legend(loc='upper left')

plt.ylim(0, 1.25)

plt.show()