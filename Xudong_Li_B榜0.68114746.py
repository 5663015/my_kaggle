import numpy as np
import pandas as pd
import tensorflow as tf
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import catboost as cbt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression, Lasso, RidgeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, roc_auc_score,log_loss, mean_absolute_error
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tqdm import tqdm
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import time
import copy
import gc
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
''''''
config_ = tf.ConfigProto()
config_.gpu_options.per_process_gpu_memory_fraction = 0.9  # 程序最多只能占用指定gpu50%的显存
config_.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.Session(config = config_)
with tf.Session(config=config_) as sess:
	pass
# -----------------------------------------------------------------------
# read data
# -----------------------------------------------------------------------
train_data = pd.read_csv('./data/first_round_training_data.csv')
test_data = pd.read_csv('./data/first_round_testing_data.csv', encoding='utf-8')
dit = {'Excellent':0,'Good':1,'Pass':2,'Fail':3}
train_data['Quality_label'] = train_data['Quality_label'].map(dit)
labels = pd.get_dummies(train_data['Quality_label']).values
submit = pd.read_csv('./data/submit_example.csv')

# 取parameter5-10，取log
features = ['Parameter5', 'Parameter6', 'Parameter7', 'Parameter8', 'Parameter9', 'Parameter10']
parameter = features + ['Parameter1', 'Parameter2', 'Parameter3', 'Parameter4']
attribute = ['Attribute1', 'Attribute2', 'Attribute3', 'Attribute4', 'Attribute5', 'Attribute6', 'Attribute7', 'Attribute8', 'Attribute9', 'Attribute10']
train_data[parameter] = np.log(train_data[parameter] + 1e-5)
train_data[attribute] = np.log(train_data[attribute] + 1e-5)
test_data[parameter] = np.log(test_data[parameter] + 1e-5)

# ------------------------------------------------------------------------
# 预测Attribute
# ------------------------------------------------------------------------
def keras_model(x, y, test_x, test_y):
	model = Sequential()
	model.add(Dense(input_shape=(x.shape[1], ), units=512, activation='tanh', kernel_regularizer=l2(5e-3)))
	#model.add(Dense(128, activation='sigmoid', kernel_regularizer=l2(5e-3)))
	model.add(Dense(1))
	model.compile(optimizer=Adam(5e-3), loss='mse', metrics=['mae'])
	reducr_lr = ReduceLROnPlateau(monitor='val_loss', patience=40, verbose=1)
	early = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
	model.fit(x, y, batch_size=256, epochs=1000, verbose=0, callbacks=[reducr_lr, early],
	          validation_data=(test_x, test_y))
	return model
	
def predict_attribute(name, x, y):
	train_data[name] = 0
	test_data[name] = 0
	kfold = KFold(n_splits=5, shuffle=True, random_state=12)
	for k, (train_index, val_index) in enumerate(kfold.split(x)):
		print('******************{}th fold********************'.format(k+1))
		train_x, train_y = x[train_index], y[train_index]
		test_x, test_y = x[val_index], y[val_index]
		# model
		# model = MLPRegressor([512], max_iter=1000, batch_size=256, learning_rate='adaptive',
		# 					 learning_rate_init=5e-3, activation='tanh',
		# 					 alpha=5e-3, early_stopping=False, n_iter_no_change=50, verbose=False)
		# model.fit(train_x, train_y)
		model = keras_model(train_x, train_y, test_x, test_y)
		# test
		train_pred = model.predict(train_x)
		print('training score: ', mean_absolute_error(y_scaler.inverse_transform(train_y), y_scaler.inverse_transform(train_pred)))
		test_pred = model.predict(test_x)
		train_data.loc[val_index, [name]] = y_scaler.inverse_transform(test_pred)
		print('testing score: ', mean_absolute_error(y_scaler.inverse_transform(test_y), y_scaler.inverse_transform(test_pred)))
		pred = model.predict(test_data[features+['Parameter1', 'Parameter2', 'Parameter3', 'Parameter4']].values).reshape((-1, ))
		test_data[name] += pred / 5.
# 预测attribute
att_features = []
for att in ['Attribute5', 'Attribute6']:#'Attribute8', 'Attribute10']:
	print('predicting {} ...'.format(att))
	name = 'pred_att{}'.format(att[-1])
	att_features.append(name)
	
	x = train_data[features + ['Parameter1', 'Parameter2', 'Parameter3', 'Parameter4']].values
	y = train_data[att].values
	x_scaler = StandardScaler()
	x = x_scaler.fit_transform(x)
	y_scaler = StandardScaler()
	y = y_scaler.fit_transform(y.reshape((-1, 1)))
	
	predict_attribute(name, x, y)
	
''''''
# 对attribute分箱
att_bins_features = []
train_temp, test_temp = [], []
for feature in att_features:
	train_bins_feature, bins = pd.qcut(train_data[feature], 10, duplicates='drop', labels=False, retbins=True)
	test_bins_feature = pd.cut(test_data[feature], bins, labels=False)
	test_bins_feature = test_bins_feature.fillna(0).astype('int64')
	
	train_data[feature + '_bins'] = train_bins_feature
	test_data[feature + '_bins'] = test_bins_feature
	
	train_data = train_data.join(pd.get_dummies(train_bins_feature, prefix=feature + '_bins'))
	test_data = test_data.join(pd.get_dummies(test_bins_feature, prefix=feature + '_bins'))
	att_bins_features.extend(pd.get_dummies(train_bins_feature, prefix=feature + '_bins').columns.values)

# ------------------------------------------------------------------------
# 特征工程
# ------------------------------------------------------------------------
# 特征分箱
fea_list = ['Parameter4', 'Parameter5', 'Parameter10']
bins_features = []
train_temp, test_temp = [], []
for feature in fea_list:
	train_bins_feature, bins = pd.qcut(train_data[feature], 10, duplicates='drop', labels=False, retbins=True)
	test_bins_feature = pd.cut(test_data[feature], bins, labels=False)
	test_bins_feature = test_bins_feature.fillna(0).astype('int64')

	train_data[feature+'_bins'] = train_bins_feature
	test_data[feature+'_bins'] = test_bins_feature
	
	train_data = train_data.join(pd.get_dummies(train_bins_feature, prefix=feature[-1]+'_bins'))
	test_data = test_data.join(pd.get_dummies(test_bins_feature, prefix=feature[-1] + '_bins'))
	bins_features.extend(pd.get_dummies(train_bins_feature, prefix=feature[-1]+'_bins').columns.values)

# 特征融合
train_data['7_8'] = train_data['Parameter7'] + train_data['Parameter8']
test_data['7_8'] = test_data['Parameter7'] + test_data['Parameter8']
train_data['7_9'] = train_data['Parameter7'] + train_data['Parameter9']
test_data['7_9'] = test_data['Parameter7'] + test_data['Parameter9']
train_data['7_10'] = train_data['Parameter7'] + train_data['Parameter10']
test_data['7_10'] = test_data['Parameter7'] + test_data['Parameter10']
train_data['8_9'] = train_data['Parameter8'] + train_data['Parameter9']
test_data['8_9'] = test_data['Parameter8'] + test_data['Parameter9']
train_data['9_10'] = train_data['Parameter9'] + train_data['Parameter10']
test_data['9_10'] = test_data['Parameter9'] + test_data['Parameter10']

train_data['9%8'] = train_data['Parameter9'] % train_data['Parameter8']
test_data['9%8'] = test_data['Parameter9'] % test_data['Parameter8']
train_data['7/8'] = train_data['Parameter7'] / train_data['Parameter8']
test_data['7/8'] = test_data['Parameter7'] / test_data['Parameter8']
train_data['7%8'] = train_data['Parameter7'] % train_data['Parameter8']
test_data['7%8'] = test_data['Parameter7'] % test_data['Parameter8']
train_data['8/9'] = train_data['Parameter8'] / train_data['Parameter9']
test_data['8/9'] = test_data['Parameter8'] / test_data['Parameter9']

scale_features = [ '7_9', '8_9', '7_8', '9_10', '7_10', '9%8', '7/8', '7%8', '8/9',]

# group特征
group_feat = []
train_data_, test_data_ = copy.deepcopy(train_data), copy.deepcopy(test_data)
for i in tqdm(['Parameter7', 'Parameter8', 'Parameter9', 'Parameter10']):
	for j in ['Parameter4', 'Parameter5']:
		if i != j:
			train_data['uni_{0}_{1}'.format(i, j)] = train_data_[i].map(train_data_.groupby(i)[j].nunique())
			test_data['uni_{0}_{1}'.format(i, j)] = test_data_[i].map(test_data_.groupby(i)[j].nunique())
			train_data['mean_{0}_{1}'.format(i, j)] = train_data_[i].map(train_data_.groupby(i)[j].mean())
			test_data['mean_{0}_{1}'.format(i, j)] = test_data_[i].map(test_data_.groupby(i)[j].mean())
			train_data['size_{0}_{1}'.format(i, j)] = train_data_[i].map(train_data_.groupby(i)[j].size())
			test_data['size_{0}_{1}'.format(i, j)] = test_data_[i].map(test_data_.groupby(i)[j].size())
			group_feat.append('uni_{0}_{1}'.format(i, j))
			group_feat.append('mean_{0}_{1}'.format(i, j))
train_data = train_data.fillna(0)
test_data = test_data.fillna(0)

# 组合所有特征
features = features +  scale_features + ['4_bins_2', '5_bins_1', '0_bins_0', 'uni_Parameter7_Parameter4', 
'mean_Parameter9_Parameter5', 'uni_Parameter7_Parameter5', 'size_Parameter7_Parameter5', 
'pred_att5_bins_8', 'pred_att5_bins_5', 'pred_att6_bins_4'] 
				
# 要去除的特征
drop_feat = ['Parameter6', '9%8', '8_9']
for f in drop_feat:
	if f in features:
		features.remove(f)
# print('*********** all train features:', features)
# 查看每个特征的nunique
# for fc in features:
# 	n = train_data['{}'.format(fc)].nunique()
# 	print(fc + ':', n)
# 查看bins_features每个特征的方差
# print('*********** std of bins features:')
# for fc in bins_features:
# 	print('{}: {}'.format(fc, train_data[fc].std()))
# 查看bins_features和label的线性关系
# print('************ corr of bins features with labels:')
# print(train_data[bins_features].corrwith(train_data['Quality_label']))
# bins_features和label的卡方检验

#pd.set_option('display.max_columns', None)
#print(train_data[features].corr())

# --------------------------------------------------------
# train
# --------------------------------------------------------
x_train = train_data[features].values
x_test = test_data[features].values

def lightgbm_model():
	model = LGBMClassifier(max_depth=5, learning_rate=0.01, n_estimators=1500, num_leaves=16,
						   objective='multiclass', silent=True,)
	return model
def xgboost_model():
	model = XGBClassifier(max_depth=5, n_estimators=800, learning_rate=0.01, silent=True,
						  objective='multi:softmax')
	return model
def catboost_model():
	cbt_model = cbt.CatBoostClassifier(#iterations=100000, learning_rate=0.01, verbose=0,
		iterations=1500, learning_rate=0.01, verbose=0, max_depth=7, task_type='GPU',
		loss_function='MultiClass')
	return cbt_model
def adaboost_model():
	ada_model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3), n_estimators=50, random_state=12)
	return ada_model

def xgboost_importance():
	model = xgboost_model()
	model.fit(x_train, np.argmax(labels, 1))
	importance_df = pd.DataFrame({'column': features, 'importance': model.feature_importances_}).sort_values(by='importance', ascending=False)
	print('************ xgboost features importance:\n', importance_df)
	return importance_df['column'].values
def catboost_importance():
	model = catboost_model()
	model.fit(x_train, np.argmax(labels, 1))
	importance = model.get_feature_importance(prettified=True)
	print('************* catboost features importance:',)
	for i in range(len(features)):
		print(features[int(importance['Feature Index'][i])] + ' :', importance['Importances'][i])


train_data['label_Excellent'] = 1 * (train_data['Quality_label'] == 0)
train_data['label_Good'] = 1 * (train_data['Quality_label'] == 1)
train_data['label_Pass'] = 1 * (train_data['Quality_label'] == 2)
train_data['label_Fail'] = 1 * (train_data['Quality_label'] == 3)
train_data['prob_Excellent'] = 0.0
train_data['prob_Good'] = 0.0
train_data['prob_Pass'] = 0.0
train_data['prob_Fail'] = 0.0
# 线下验证group 连续构造
for group in range(50):
	train_data['group_%s' % group] = (train_data.index + group) // 50 % 120
# 线下验证group 随机构造
for group in range(50, 100):
	name = 'group_%s' % group
	train_data[name] = 0
	kfold = KFold(n_splits=120, shuffle=True, random_state=group)
	split = kfold.split(train_data)
	i = 0
	for i, (train_index, valid_index) in enumerate(split):
		train_data.iloc[valid_index, -1] = i


catboost_importance()
xgboost_importance()
lr = LogisticRegression()
# k-fold train
def kfold_train(mode):
	acc_list, loss_list = [], []
	prediction = np.zeros((x_test.shape[0], 4))
	result_list = []
	n = 10
	for i in range(n):
		print(str(i+1) + ' th kflod' + '*'*50)
		result = []
		kf = KFold(n_splits=5, shuffle=True, random_state=i)
		kfold_list = []
		for k, (train_index, test_index) in enumerate(kf.split(x_train)):
			print(str(k+1) + 'fold--------------')
			train_x, train_y = x_train[train_index], labels[train_index]
			test_x, test_y = x_train[test_index], labels[test_index]
			# train
			if mode == 'cat':
				model = catboost_model()
				model.fit(train_x, np.argmax(train_y, 1), eval_set=(test_x, np.argmax(test_y, 1)),
					  #early_stopping_rounds=1000, verbose=False
						)
				#print(pd.DataFrame({'column': features, 'importance': model.feature_importances_}).sort_values(by='importance'))
			if mode == 'lgb':
				model = lightgbm_model()
				model.fit(train_x, np.argmax(train_y, 1), eval_set=(test_x, np.argmax(test_y, 1)),
					  # early_stopping_rounds=50, verbose=True
						  verbose=False
						  )
			if mode == 'xgb':
				model = xgboost_model()
				model.fit(train_x, np.argmax(train_y, 1), verbose=True)
			if mode == 'stack':
				model = StackingClassifier(
					classifiers=[catboost_model(), lightgbm_model(), xgboost_model(),
								 adaboost_model()],
					use_probas=True, average_probas=False, meta_classifier=lr)
				model.fit(train_x, np.argmax(train_y, 1))
			# test
			pred = model.predict_proba(test_x)
			acc = accuracy_score(np.argmax(test_y, 1), np.argmax(pred, 1))
			loss = log_loss(test_y, pred)
			acc_list.append(acc)
			loss_list.append(loss)
			kfold_list.append(loss)
			print('test acc: %f, test loss: %f' % (acc, loss))
			# 用于线下验证
			X_valid = train_data.iloc[test_index, :].copy()
			X_valid.loc[:, ['prob_Excellent', 'prob_Good', 'prob_Pass', 'prob_Fail']] = pred
			result.append(X_valid)
			# predict
			prediction += model.predict_proba(x_test)
		print('this fold mean loss:', np.mean(kfold_list))
		result_list.append(pd.concat(result))
	print('*'*50)
	print('mean acc: %f, mean loss: %f' % (np.mean(acc_list), np.mean(loss_list)))
	prediction = prediction / (5. * n)
	# 线下评估
	mean = []
	for group in range(100):
		for result in result_list:
			temp = result.groupby(['group_%s' % group], as_index=False)[
				'prob_Excellent', 'prob_Good', 'prob_Pass', 'prob_Fail', 'label_Excellent', 'label_Good', 'label_Pass', 'label_Fail'].mean()
			a = np.abs(temp.loc[:, ['prob_Excellent', 'prob_Good', 'prob_Pass', 'prob_Fail']].values
					   - temp.loc[:, ['label_Excellent', 'label_Good', 'label_Pass', 'label_Fail']].values).mean()
			mean.append(1 / (1 + 10 * a))
	print("线下mae评估：", np.mean(mean), np.std(mean))

	return prediction

def submit_result(prediction):
	sub = test_data[['Group']]
	prob_cols = [i for i in submit.columns if i not in ['Group']]
	for i, f in enumerate(prob_cols):
		sub[f] = prediction[:, i]
	for i in prob_cols:
		sub[i] = sub.groupby('Group')[i].transform('mean')
	sub = sub.drop_duplicates()
	sub.to_csv("submission2.csv", index=False)

time1 = time.clock()
prediction = kfold_train('stack')
time2 = time.clock()
print('running time: ', str((time2 - time1)/60))
submit_result(prediction)

'''
************* catboost features importance:
Parameter5 : 11.20118455850489
Parameter10 : 10.749357860717216
a_bins_9 : 9.842048872819063   x
7_10 : 7.806399754704091
9_10 : 6.999432306848506
7/8 : 6.480710520544819
8/9 : 3.766202907363485
7_9 : 3.7069756798180467
7%8 : 3.3616155956668043
7_8 : 2.7760587349727306
Parameter8 : 2.5519225175123386
a_bins_8 : 2.3715868649165572     ~
5_bins_1 : 2.352172308707592
Parameter9 : 2.1365812960133677
Parameter7 : 1.6863220260037806
0_bins_0 : 1.3940539000550543
4_bins_2 : 1.126198277437627
a_bins_5 : 1.0857627419475253    x
a_bins_7 : 0.9287459824609547    x
4_bins_3 : 0.9049138466672559
a_bins_6 : 0.8785405182549512    x
4_bins_6 : 0.8366744375690117
a_bins_3 : 0.823741591470417
4_bins_5 : 0.8205452685511854
4_bins_4 : 0.8135620648209294
4_bins_0 : 0.7916034598354323
a_bins_0 : 0.766674136639238
a_bins_4 : 0.762410711372699
4_bins_7 : 0.6725345099372022
4_bins_8 : 0.6638588739996741
5_bins_5 : 0.6590578546046914
5_bins_7 : 0.6528514821657395
0_bins_9 : 0.5906402501532555
4_bins_1 : 0.5777114679901706
0_bins_7 : 0.5740982298719612
0_bins_5 : 0.5581238137145709
4_bins_9 : 0.5396953666237534
0_bins_1 : 0.47056909890455006
5_bins_6 : 0.46283606204591343
0_bins_2 : 0.4093845760748505
a_bins_1 : 0.3867385534590077
a_bins_2 : 0.3834851739311365
0_bins_4 : 0.3805205105572269
5_bins_4 : 0.36675319809763574
5_bins_8 : 0.3409598395387928
0_bins_8 : 0.33671728812341895
5_bins_2 : 0.32089129025143037
5_bins_3 : 0.2765247693468506
0_bins_6 : 0.2459193014702636
0_bins_3 : 0.24267943929695798
5_bins_0 : 0.0952813595535202
5_bins_9 : 0.0701689480918316
************ xgboost features importance:
          column  importance
0    Parameter5    0.205904
4   Parameter10    0.117866
7          9_10    0.091564
8          7_10    0.081723
5           7_9    0.036351
10          7%8    0.024373
11          8/9    0.023595
14     4_bins_2    0.023040
9           7/8    0.020833
2    Parameter8    0.019681
16     4_bins_4    0.019681
15     4_bins_3    0.017266
18     4_bins_6    0.016989
12     4_bins_0    0.016086
45     a_bins_3    0.015406
19     4_bins_7    0.014865
50     a_bins_8    0.014782
6           7_8    0.013477
49     a_bins_7    0.013422
20     4_bins_8    0.013172
17     4_bins_5    0.013005
21     4_bins_9    0.012728
13     4_bins_1    0.012186
51     a_bins_9    0.012006
27     5_bins_5    0.009632
1    Parameter7    0.009369
39     0_bins_7    0.009272
46     a_bins_4    0.009133
25     5_bins_3    0.008633
44     a_bins_2    0.008591
47     a_bins_5    0.008425
48     a_bins_6    0.008078
29     5_bins_7    0.007023
3    Parameter9    0.006523
37     0_bins_5    0.006523
36     0_bins_4    0.005913
42     a_bins_0    0.005857
23     5_bins_1    0.005816
28     5_bins_6    0.004983
33     0_bins_1    0.004900
26     5_bins_4    0.004580
43     a_bins_1    0.004539
38     0_bins_6    0.004247
30     5_bins_8    0.004094
34     0_bins_2    0.004025
35     0_bins_3    0.003720
40     0_bins_8    0.003289
24     5_bins_2    0.002831
41     0_bins_9    0.000000
32     0_bins_0    0.000000
31     5_bins_9    0.000000
22     5_bins_0    0.000000
'''

'''
************* catboost features importance:
Parameter5 : 13.6738405309796
Parameter10 : 12.762812995935567
pred_att4_bins_9 : 8.224181410533244
Parameter8 : 6.152513940964753
Parameter9 : 5.454616468881103
Parameter7 : 4.092396280359963
pred_att8_bins_9 : 2.49410010425532
pred_att6_bins_9 : 2.2759421495723147
pred_att4_bins_8 : 2.251034135241305
pred_att5_bins_9 : 2.0602319710417154
pred_att5_bins_0 : 1.4162151328072419
pred_att4_bins_0 : 1.3683731322356292
pred_att4_bins_7 : 1.3232988506205383
pred_att5_bins_8 : 1.2558425948267393
pred_att7_bins_9 : 1.2260779661397723
pred_att4_bins_6 : 1.1067840913603904
pred_att5_bins_1 : 1.1062978937398007
pred_att6_bins_8 : 1.0991878216049726
pred_att7_bins_3 : 0.992770631224262
pred_att7_bins_8 : 0.9900684574344361
pred_att0_bins_6 : 0.9181561400393718
pred_att7_bins_2 : 0.9075475490583366
pred_att5_bins_5 : 0.8621695484414696
pred_att8_bins_0 : 0.8593277659986472
pred_att8_bins_3 : 0.8561131457163846
pred_att0_bins_4 : 0.845397838057322
pred_att4_bins_5 : 0.8411847116040937
pred_att8_bins_2 : 0.8337224910825118
pred_att0_bins_9 : 0.8254167697380879
pred_att5_bins_3 : 0.7877177021934337
pred_att6_bins_5 : 0.7809723872223776
pred_att0_bins_5 : 0.774628549648218
pred_att5_bins_6 : 0.736960372143601
pred_att0_bins_3 : 0.717064677971409
pred_att7_bins_5 : 0.7136656766716013
pred_att8_bins_7 : 0.7128036537716315
pred_att6_bins_3 : 0.7103388303703998
pred_att8_bins_1 : 0.7090159454737434
pred_att5_bins_4 : 0.6555957283930937
pred_att5_bins_7 : 0.6422315630943074
pred_att8_bins_6 : 0.6360279978706571
pred_att8_bins_8 : 0.6279098481251498
pred_att6_bins_1 : 0.6273785581617839
pred_att8_bins_4 : 0.6186895093598641
pred_att6_bins_6 : 0.5964509476994152
pred_att0_bins_8 : 0.5904723128308289
pred_att4_bins_1 : 0.5870293987053518
pred_att0_bins_2 : 0.5846836315799356
pred_att7_bins_7 : 0.5815358582798338
pred_att7_bins_0 : 0.5652696501024731
pred_att4_bins_3 : 0.5622904795817256
pred_att7_bins_1 : 0.54234382082694
pred_att0_bins_7 : 0.5412794471972454
pred_att5_bins_2 : 0.5279203342804898
pred_att4_bins_4 : 0.5264363929505507
pred_att7_bins_6 : 0.5173973884323524
pred_att6_bins_4 : 0.5132947092610187
pred_att8_bins_5 : 0.5069587598890898
pred_att6_bins_7 : 0.49592955921536114
pred_att0_bins_0 : 0.450051243203849
pred_att7_bins_4 : 0.3968071908370579
pred_att6_bins_2 : 0.380419155854426
pred_att0_bins_1 : 0.35722399371092706
pred_att6_bins_0 : 0.34086151928766417
pred_att4_bins_2 : 0.30872068630731514
'''
'''
************* catboost features importance:
Parameter10 : 12.421669073760802
pred_att5_bins_9 : 12.117051613183358
Parameter5 : 11.589649012367182
7_10 : 8.06917039211041
9_10 : 6.730762204397951
7/8 : 5.810785257462321
8/9 : 3.076049634846653
7_9 : 2.688513551684355
7%8 : 2.633486883408503
5_bins_1 : 2.391103835877468
4_bins_2 : 2.3592215069686127
pred_att6_bins_9 : 2.2707475496169893
7_8 : 2.209967256907064
Parameter8 : 1.8710062824258011
mean_Parameter9_Parameter5 : 1.774311557762499
Parameter9 : 1.6752547095985761
uni_Parameter7_Parameter5 : 1.4806578098715824
pred_att5_bins_8 : 1.4030258111789191
pred_att5_bins_6 : 1.3956743549843273   x
pred_att5_bins_7 : 1.2869422459562987   x
pred_att6_bins_7 : 1.0522229259654272
pred_att5_bins_4 : 1.0200999110454798
Parameter7 : 0.9850398671210224
pred_att6_bins_1 : 0.941557571938989
pred_att5_bins_0 : 0.9309387137676998
size_Parameter7_Parameter5 : 0.8932195762289393
pred_att5_bins_3 : 0.8316398299096178
pred_att6_bins_3 : 0.8079302758486181
0_bins_0 : 0.7872712186244709
pred_att6_bins_5 : 0.7840420430509859
pred_att6_bins_6 : 0.7750278698477823
pred_att5_bins_5 : 0.7567085578016477
pred_att5_bins_8 : 0.7306293920593866
pred_att6_bins_4 : 0.5919479380366287
uni_Parameter7_Parameter4 : 0.5702408010519732
pred_att6_bins_8 : 0.5674971607206163
pred_att5_bins_2 : 0.521665470121734
pred_att5_bins_1 : 0.5045367212198079
pred_att6_bins_2 : 0.4830935100890431
pred_att6_bins_0 : 0.2096401011804864
************ xgboost features importance:
                         column  importance
0                   Parameter5    0.226931
4                  Parameter10    0.134889
7                         9_10    0.106265
8                         7_10    0.092221
12                    4_bins_2    0.028788
5                          7_9    0.027883
11                         8/9    0.024646
9                          7/8    0.023425
10                         7%8    0.022219
27            pred_att5_bins_7    0.021931
19            pred_att5_bins_8    0.019119
23            pred_att5_bins_3    0.017720
2                   Parameter8    0.017651
29            pred_att5_bins_9    0.016033
34            pred_att6_bins_4    0.015347
37            pred_att6_bins_7    0.014250
1                   Parameter7    0.014154
16  mean_Parameter9_Parameter5    0.012824
6                          7_8    0.012618
25            pred_att5_bins_5    0.012275
33            pred_att6_bins_3    0.011699
15   uni_Parameter7_Parameter4    0.011507
24            pred_att5_bins_4    0.011438
35            pred_att6_bins_5    0.010149
31            pred_att6_bins_1    0.009985
22            pred_att5_bins_2    0.009738
36            pred_att6_bins_6    0.009436
38            pred_att6_bins_8    0.009025
21            pred_att5_bins_1    0.008147
32            pred_att6_bins_2    0.007639
20            pred_att5_bins_0    0.007447
26            pred_att5_bins_6    0.007406
39            pred_att6_bins_9    0.007173
3                   Parameter9    0.007132
13                    5_bins_1    0.006679
17   uni_Parameter7_Parameter5    0.002181
30            pred_att6_bins_0    0.002030
28            pred_att5_bins_8    0.000000
18  size_Parameter7_Parameter5    0.000000
14                    0_bins_0    0.000000
'''
