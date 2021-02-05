#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lifelines import KaplanMeierFitter
from matplotlib import pyplot as plt
from lifelines.utils import median_survival_times
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
from lifelines import CoxPHFitter
from pysurvival.utils.display import correlation_matrix
from sklearn.model_selection import GridSearchCV
from pysurvival.models.semi_parametric import NonLinearCoxPHModel
from pysurvival.utils.sklearn_adapter import sklearn_adapter
from pysurvival.utils.metrics import concordance_index
from pysurvival.utils.display import integrated_brier_score
from pysurvival.utils.metrics import brier_score
from pysurvival.utils.display import compare_to_actual


# In[2]:


data1_columns = ['ID', 'sex', 'age', 'T', 'N', 'stage', 'pathology', 'radiation_dose', 'RT_duration', 'neoadjuvantCT', 
                 'neoadjuvantCT_regime', 'concurrentCT', 'adjuvantCT', 'adjuvantCT_regime', 'targeted_therapy', 
                 'targeted_therapy_regime', 'mucositis', 'RT_startdate', 'last_followup', 'disease_progress', 'PFS']
data2_columns = ['ID', 'pre_weight', 'post_weight', 'pre_albumin', 'post_albumin', 'pre_haemoglobin', 'post_haemoglobin', 
                 'pre_neutrophil', 'post_neutrophil', 'pre_platelet', 'post_platelet', 'pre_lymphocyte', 'post_lymphocyte']
final_list = ['ID', 'sex', 'age', 'T', 'N', 'stage', 'pathology', 'radiation_dose', 'RT_duration', 'neoadjuvantCT', 
              'neoadjuvantCT_regime', 'concurrentCT', 'adjuvantCT', 'adjuvantCT_regime', 'targeted_therapy', 
              'targeted_therapy_regime', 'mucositis', 'followup_length', 'weight_change', 'albumin_change', 'haemoglobin_change', 
              'neutrophil_change', 'platelet_change', 'lymphocyte_change', 'disease_progress', 'PFS']
features = ['sex', 'age', 'T', 'N', 'pathology', 'radiation_dose', 'RT_duration', 'neoadjuvantCT', 'concurrentCT', 
            'adjuvantCT']
predictors_1 = ['sex', 'age', 'T', 'N', 'pathology', 'radiation_dose', 'RT_duration', 'neoadjuvantCT', 'concurrentCT', 
                'adjuvantCT', 'PFS', 'disease_progress']
predictors_2 = ['sex', 'age', 'T', 'N', 'pathology', 'radiation_dose', 'RT_duration', 'neoadjuvantCT_regime', 'concurrentCT', 
                'adjuvantCT_regime', 'mucositis', 'weight_change', 'albumin_change', 'haemoglobin_change', 'PFS', 
                'disease_progress']
nutritional_info = ['weight_change', 'albumin_change', 'haemoglobin_change']
haematological_info = ['pre_neutrophil', 'pre_platelet', 'pre_lymphocyte']
Y = ['PFS', 'disease_progress']
sex = ['sex', 'PFS', 'disease_progress']
age = ['age', 'PFS', 'disease_progress']
T = ['T', 'PFS', 'disease_progress']
N = ['N', 'PFS', 'disease_progress']
pathology = ['pathology', 'PFS', 'disease_progress']
radiation_dose = ['radiation_dose', 'PFS', 'disease_progress']
RT_duration = ['RT_duration', 'PFS', 'disease_progress']
neoadjuvantCT = ['neoadjuvantCT', 'PFS', 'disease_progress']
concurrentCT = ['concurrentCT', 'PFS', 'disease_progress']
adjuvantCT = ['adjuvantCT', 'PFS', 'disease_progress']
weight_change = ['weight_change', 'PFS', 'disease_progress']
albumin_change = ['albumin_change', 'PFS', 'disease_progress']
haemoglobin_change = ['haemoglobin_change', 'PFS', 'disease_progress']
pre_neutrophil = ['pre_neutrophil', 'PFS', 'disease_progress']
pre_platelet = ['pre_platelet', 'PFS', 'disease_progress']
pre_lymphocyte = ['pre_lymphocyte', 'PFS', 'disease_progress']


# In[3]:


data1 = pd.read_csv('Data1.csv')[data1_columns]
data2 = pd.read_csv('Data2.csv')[data2_columns]


# In[4]:


data1['PFS'].hist(bins=np.arange(start=16, stop=100, step=2), figsize=[14,6])


# In[5]:


data1['pathology'].value_counts()


# In[6]:


data1['pathology'] = data1['pathology'].apply(lambda x: 2 if x == str(3) else 3)


# In[7]:


data1.isna().any()


# In[8]:


data1['neoadjuvantCT_regime'] = data1['neoadjuvantCT_regime'].fillna(0)
data1['adjuvantCT_regime'] = data1['adjuvantCT_regime'].fillna(0)
data1['targeted_therapy_regime'] = data1['targeted_therapy_regime'].fillna(0)
data1.isna().any()


# In[9]:


data1['RT_startdate'] = pd.to_datetime(data1['RT_startdate'])
data1['last_followup'] = pd.to_datetime(data1['last_followup'])

data1['followup_length'] = data1['last_followup'] - data1['RT_startdate']


# In[10]:


data2.isna().any()


# In[11]:


def compute_change(x, y):
    return ((y - x) / x) * 100


# In[12]:


data2['weight_change'] = compute_change(data2['pre_weight'], data2['post_weight'])
data2['albumin_change'] = compute_change(data2['pre_albumin'], data2['post_albumin'])
data2['haemoglobin_change'] = compute_change(data2['pre_haemoglobin'], data2['post_haemoglobin'])
data2['neutrophil_change'] = compute_change(data2['pre_neutrophil'], data2['post_neutrophil'])
data2['platelet_change'] = compute_change(data2['pre_platelet'], data2['post_platelet'])
data2['lymphocyte_change'] = compute_change(data2['pre_lymphocyte'], data2['post_lymphocyte'])


# In[13]:


data2[['pre_weight', 'post_weight', 'weight_change']]


# In[14]:


df = data1.join(data2.set_index('ID'), on = 'ID')[final_list]


# In[15]:


df


# In[ ]:





# In[16]:


SEED = 0


# In[17]:


data_positive = df[df.disease_progress == 1]
data_negative = df[df.disease_progress == 0]


# In[18]:


data_positive['PFS_binned'] = pd.qcut(data_positive['PFS'], q=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], labels=False)


# In[19]:


data_negative['PFS_binned'] = pd.qcut(data_negative['PFS'], q=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], labels=False)


# In[20]:


train1, test1 = train_test_split(data_positive[final_list], test_size=0.1, stratify = data_positive['PFS_binned'], random_state = 0)
train2, test2 = train_test_split(data_negative[final_list], test_size=0.1, stratify = data_negative['PFS_binned'], random_state = 0)


# In[21]:


train = pd.concat([train1, train2])
test = pd.concat([test1, test2])


# In[22]:


data1['PFS'].hist(bins=np.arange(start=16, stop=100, step=2), figsize=[14,6])


# In[23]:


train['PFS'].hist(bins=np.arange(start=16, stop=100, step=2), figsize=[14,6])


# In[24]:


test['PFS'].hist(bins=np.arange(start=16, stop=100, step=2), figsize=[14,6])


# In[ ]:





# In[25]:


kmf_train = KaplanMeierFitter()
kmf_test = KaplanMeierFitter()


# In[26]:


figure = plt.figure(figsize = (12, 8), tight_layout = False)

ax = figure.add_subplot(111)

t = np.linspace(0, 84, 85)

kmf_train.fit(train['PFS'], event_observed=train['disease_progress'], timeline=t, label='Train set')
ax = kmf_train.plot_survival_function(show_censors=True, ci_force_lines=False, ci_show=False, ax=ax)

kmf_test.fit(test['PFS'], event_observed=test['disease_progress'], timeline=t, label='Test set')
ax = kmf_test.plot_survival_function(show_censors=True, ci_force_lines=False, ci_show=False, ax=ax)

add_at_risk_counts(kmf_train, kmf_test, ax=ax, fontsize=12)

ax.set_xlabel('Time (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Survival probability', fontsize=12, fontweight='bold')
ax.legend(fontsize=12)

ax.text(10, 0.75, 'p=0.85', fontsize=12, fontweight='bold')


# In[27]:


print(kmf_train.median_survival_time_)
print(kmf_test.median_survival_time_)


# In[28]:


print(median_survival_times(kmf_train.confidence_interval_))
print(median_survival_times(kmf_test.confidence_interval_))


# In[29]:


print(kmf_train.event_table)
print(kmf_test.event_table)


# In[30]:


print('Survival probability for t=60 for train set: ', kmf_train.predict(60))
print('Survival probability for t=60 for test set: ', kmf_test.predict(60))


# In[31]:


results = logrank_test(train['PFS'], test['PFS'], train['disease_progress'], test['disease_progress'], alpha=.95)

results.print_summary()


# In[ ]:





# In[32]:


cph = CoxPHFitter()
data = df[predictors_1]

cph.fit(data, 'PFS', event_col='disease_progress')
cph.print_summary()


# In[33]:


cph = CoxPHFitter()
data = train[sex]
cph.fit(data, 'PFS', event_col='disease_progress')
cph.print_summary()

cph = CoxPHFitter()
data = train[age]
cph.fit(data, 'PFS', event_col='disease_progress')
cph.print_summary()

cph = CoxPHFitter()
data = train[T]
cph.fit(data, 'PFS', event_col='disease_progress')
cph.print_summary()

cph = CoxPHFitter()
data = train[N]
cph.fit(data, 'PFS', event_col='disease_progress')
cph.print_summary()

cph = CoxPHFitter()
data = train[pathology]
cph.fit(data, 'PFS', event_col='disease_progress')
cph.print_summary()

cph = CoxPHFitter()
data = train[radiation_dose]
cph.fit(data, 'PFS', event_col='disease_progress')
cph.print_summary()

cph = CoxPHFitter()
data = train[RT_duration]
cph.fit(data, 'PFS', event_col='disease_progress')
cph.print_summary()

cph = CoxPHFitter()
data = train[neoadjuvantCT]
cph.fit(data, 'PFS', event_col='disease_progress')
cph.print_summary()

cph = CoxPHFitter()
data = train[concurrentCT]
cph.fit(data, 'PFS', event_col='disease_progress')
cph.print_summary()

cph = CoxPHFitter()
data = train[adjuvantCT]
cph.fit(data, 'PFS', event_col='disease_progress')
cph.print_summary()

cph = CoxPHFitter()
data = train[weight_change]
cph.fit(data, 'PFS', event_col='disease_progress')
cph.print_summary()

cph = CoxPHFitter()
data = train[albumin_change]
cph.fit(data, 'PFS', event_col='disease_progress')
cph.print_summary()

cph = CoxPHFitter()
data = train[haemoglobin_change]
cph.fit(data, 'PFS', event_col='disease_progress')
cph.print_summary()


# In[ ]:





# In[34]:


correlation_matrix(df[features], figure_size=(30,15), text_fontsize=10)


# In[ ]:





# In[35]:


def compute_scores(model, table, timepoints, variables):
    c_indexes = []
    for i in timepoints:
        table.loc[:, 'disease_progress_temp'] = table['disease_progress']
        table.loc[:, 'PFS_temp'] = table['PFS']
        table.loc[table.PFS > i, 'disease_progress_temp'] = 0
        table.loc[table.PFS > i, 'PFS_temp'] = i
        c_indexes.append(concordance_index(model, table[variables], table['PFS_temp'], table['disease_progress_temp']))
    brier_scores = brier_score(model, table[variables], table['PFS'], table['disease_progress'], t_max=84, figure_size=(20, 6.5))
    return c_indexes, brier_scores


# In[ ]:





# In[36]:


nlcph = NonLinearCoxPHModel(structure=[{'activation': 'ReLU', 'num_units': 64}], auto_scaler=True)
nlcph.fit(train[features], train['PFS'], train['disease_progress'], init_method='glorot_uniform', lr=0.0001)

c_index = concordance_index(nlcph, test[features], test['PFS'], test['disease_progress'])
print('C-index: {:.2f}'.format(c_index))

ibs = integrated_brier_score(nlcph, test[features], test['PFS'], test['disease_progress'], t_max=84, 
                             figure_size=(20, 6.5))
print('IBS: {:.2f}'.format(ibs))

results = compare_to_actual(nlcph, test[features], test['PFS'], test['disease_progress'],
                            is_at_risk = False,  figure_size=(16, 6),
                            metrics = ['rmse', 'mean', 'median'])

nlcph_c_index, nlcph_brier_score = compute_scores(nlcph, test, list(np.arange(0, 86, 2)), features)

nlcph_c_index_table_nonutrition = pd.DataFrame(columns=list(np.arange(0, 86, 2)))
series_c_index = pd.Series(nlcph_c_index, index = nlcph_c_index_table_nonutrition.columns)
nlcph_c_index_table_nonutrition = nlcph_c_index_table_nonutrition.append(series_c_index, ignore_index=True)
nlcph_brier_score_table_nonutrition = pd.DataFrame(columns=nlcph_brier_score[0])
series_brier_score = pd.Series(nlcph_brier_score[1], index = nlcph_brier_score_table_nonutrition.columns)
nlcph_brier_score_table_nonutrition = nlcph_brier_score_table_nonutrition.append(series_brier_score, ignore_index=True)

nlcph_c_index_table_nonutrition.to_csv('nlcph_c_index_nonutrition.csv', index=None)
nlcph_brier_score_table_nonutrition.to_csv('nlcph_brier_score_nonutrition.csv', index=None)


# In[37]:


survival_pred_nlcph_nonutrition = {}
for i in range(15):
    survival_pred_nlcph_nonutrition[str(i)] = []
    for j in list(np.arange(0, 73, 1)):
        survival_pred_nlcph_nonutrition[str(i)].append(nlcph.predict_survival(test.iloc[i, :][features], t=j))


# In[38]:


x = list(np.arange(0, 73, 1))

plt.figure(figsize=(20,10))
for i in range(15):
    plt.plot(x, survival_pred_nlcph_nonutrition[str(i)], label=('P'+str(i+1)))
plt.xlim([0, 72])
plt.xlabel('Progression-free survival time (months)', fontsize=12, fontweight='bold')
plt.ylabel('Survival probability based on DeepSurv model', fontsize=12, fontweight='bold')
plt.legend(fontsize=12)


# In[39]:


risk_prediction_nonutrition_table = pd.DataFrame(data=nlcph.predict_risk(test[features]), columns=['risk_prediction'])
risk_prediction_nonutrition_table.to_csv('nlcph_risk_prediction_nonutrition.csv', index=None)

test_nonutrition = test.copy()
test_nonutrition['risk_prediction'] =  nlcph.predict_risk(test[features])

median = np.median(test_nonutrition['risk_prediction'])

low_risk_group = test_nonutrition[test_nonutrition.risk_prediction <= median]
high_risk_group = test_nonutrition[test_nonutrition.risk_prediction > median]

kmf_low_risk = KaplanMeierFitter()
kmf_high_risk = KaplanMeierFitter()

figure = plt.figure(figsize = (12, 8), tight_layout = False)
ax = figure.add_subplot(111)
t = np.linspace(0, 84, 85)

kmf_low_risk.fit(low_risk_group['PFS'], event_observed=low_risk_group['disease_progress'], timeline=t, label='Low-risk group')
ax = kmf_low_risk.plot_survival_function(show_censors=True, ci_force_lines=False, ci_show=False, ax=ax)

kmf_high_risk.fit(high_risk_group['PFS'], event_observed=high_risk_group['disease_progress'], timeline=t, label='High-risk group')
ax = kmf_high_risk.plot_survival_function(show_censors=True, ci_force_lines=False, ci_show=False, ax=ax)

add_at_risk_counts(kmf_low_risk, kmf_high_risk, ax=ax, fontsize=12)

ax.set_xlabel('Time (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Survival probability based on DeepSurv model', fontsize=12, fontweight='bold')
ax.legend(fontsize=12)

ax.text(10, 0.65, 'p=0.07', fontsize=12, fontweight='bold')


# In[40]:


results_low_high_risk = logrank_test(low_risk_group['PFS'], high_risk_group['PFS'], low_risk_group['disease_progress'], high_risk_group['disease_progress'], alpha=.95)

results_low_high_risk.print_summary()


# In[ ]:





# In[41]:


nlcph = NonLinearCoxPHModel(structure=[{'activation': 'ReLU', 'num_units': 64}], auto_scaler=True)
nlcph.fit(train[features + nutritional_info], train['PFS'], train['disease_progress'], init_method='glorot_uniform', lr=0.0001)

c_index = concordance_index(nlcph, test[features + nutritional_info], test['PFS'], test['disease_progress'])
print('C-index: {:.2f}'.format(c_index))

ibs = integrated_brier_score(nlcph, test[features + nutritional_info], test['PFS'], test['disease_progress'], t_max=84, 
                             figure_size=(20, 6.5))
print('IBS: {:.2f}'.format(ibs))

results = compare_to_actual(nlcph, test[features + nutritional_info], test['PFS'], test['disease_progress'],
                            is_at_risk = False,  figure_size=(16, 6),
                            metrics = ['rmse', 'mean', 'median'])

nlcph_c_index, nlcph_brier_score = compute_scores(nlcph, test, list(np.arange(0, 86, 2)), features + nutritional_info)

nlcph_c_index_table_nutrition = pd.DataFrame(columns=list(np.arange(0, 86, 2)))
series_c_index = pd.Series(nlcph_c_index, index = nlcph_c_index_table_nutrition.columns)
nlcph_c_index_table_nutrition = nlcph_c_index_table_nutrition.append(series_c_index, ignore_index=True)
nlcph_brier_score_table_nutrition = pd.DataFrame(columns=nlcph_brier_score[0])
series_brier_score = pd.Series(nlcph_brier_score[1], index = nlcph_brier_score_table_nutrition.columns)
nlcph_brier_score_table_nutrition = nlcph_brier_score_table_nutrition.append(series_brier_score, ignore_index=True)

nlcph_c_index_table_nutrition.to_csv('nlcph_c_index_nutrition.csv', index=None)
nlcph_brier_score_table_nutrition.to_csv('nlcph_brier_score_nutrition.csv', index=None)


# In[42]:


survival_pred_nlcph_nutrition = {}
for i in range(15):
    survival_pred_nlcph_nutrition[str(i)] = []
    for j in list(np.arange(0, 73, 1)):
        survival_pred_nlcph_nutrition[str(i)].append(nlcph.predict_survival(test.iloc[i, :][features + nutritional_info], t=j))


# In[43]:


x = list(np.arange(0, 73, 1))

plt.figure(figsize=(20,10))
for i in range(15):
    plt.plot(x, survival_pred_nlcph_nutrition[str(i)], label=('P'+str(i+1)))
plt.xlim([0, 72])
plt.xlabel('Progression-free survival time (months)', fontsize=12, fontweight='bold')
plt.ylabel('Survival probability based on DeepSurv model', fontsize=12, fontweight='bold')
plt.legend(fontsize=12)


# In[44]:


risk_prediction_nutrition_table = pd.DataFrame(data=nlcph.predict_risk(test[features + nutritional_info]), columns=['risk_prediction'])
risk_prediction_nutrition_table.to_csv('nlcph_risk_prediction_nutrition.csv', index=None)

test_nutrition = test.copy()
test_nutrition['risk_prediction'] =  nlcph.predict_risk(test[features + nutritional_info])

median = np.median(test_nutrition['risk_prediction'])

low_risk_group = test_nutrition[test_nutrition.risk_prediction <= median]
high_risk_group = test_nutrition[test_nutrition.risk_prediction > median]

kmf_low_risk = KaplanMeierFitter()
kmf_high_risk = KaplanMeierFitter()

figure = plt.figure(figsize = (12, 8), tight_layout = False)
ax = figure.add_subplot(111)
t = np.linspace(0, 84, 85)

kmf_low_risk.fit(low_risk_group['PFS'], event_observed=low_risk_group['disease_progress'], timeline=t, label='Low-risk group')
ax = kmf_low_risk.plot_survival_function(show_censors=True, ci_force_lines=False, ci_show=False, ax=ax)

kmf_high_risk.fit(high_risk_group['PFS'], event_observed=high_risk_group['disease_progress'], timeline=t, label='High-risk group')
ax = kmf_high_risk.plot_survival_function(show_censors=True, ci_force_lines=False, ci_show=False, ax=ax)

add_at_risk_counts(kmf_low_risk, kmf_high_risk, ax=ax, fontsize=12)

ax.set_xlabel('Time (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Survival probability based on DeepSurv model', fontsize=12, fontweight='bold')
ax.legend(fontsize=12)

ax.text(10, 0.65, 'p=0.02', fontsize=12, fontweight='bold')


# In[45]:


results_low_high_risk = logrank_test(low_risk_group['PFS'], high_risk_group['PFS'], low_risk_group['disease_progress'], high_risk_group['disease_progress'], alpha=.95)

results_low_high_risk.print_summary()


# In[ ]:




