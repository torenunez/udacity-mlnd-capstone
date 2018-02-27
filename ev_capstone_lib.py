import math
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans


def pivot_raw_df(df, metric):

	df_pivot = pd.wide_to_long(df, stubnames='Interval_', i='House ID', j=metric).reset_index()
	df_pivot.columns = ['House ID', 'Interval', metric]
	df_pivot['Interval'] = pd.to_numeric(df_pivot['Interval'], downcast='integer')
	df_pivot = df_pivot.sort_values(['House ID', 'Interval']).reset_index(drop=True)

	return df_pivot


def add_temp_dim(df):

	length = len(df['House ID'].unique())

	temp_dim_dict = {
	"Day": np.tile(np.repeat(np.arange(1,61),48),length),
	"Hour": np.tile(np.tile(np.repeat(np.arange(1,25),2),60),length),
	"Half Hour": np.tile(np.tile(np.arange(1,49),60),length)
	}

	temp_dim_df = pd.DataFrame.from_dict(temp_dim_dict)
	df_tmp = pd.concat([df, temp_dim_df], axis=1)

	old_cols = df_tmp.columns.values.tolist()
	col_order = ['House ID','Day','Hour','Half Hour','Interval', 'kWh', 'Label']

	new_cols = [c for c in col_order if c in old_cols]
	df_tmp_final = df_tmp[new_cols] 

	return df_tmp_final


def fill_with_mean(df):

	df_filled = df.copy()

	all_missing_rows = df['kWh'].isnull()

	mean_df = df.groupby(['House ID','Half Hour']).agg({'kWh':'mean'}).reset_index()

	# a merge was taking too much memory
	# looping over missing values takes less memory
	for idx, row in df[all_missing_rows].iterrows():

		try:
			missing_interval = row['Interval']
			missing_house = row['House ID']
			missing_half_hour = row['Half Hour']

			house_rm = mean_df['House ID'] == missing_house
			half_hour_rm = mean_df['Half Hour'] == missing_half_hour

			house_half_hour_average = mean_df[house_rm&half_hour_rm]['kWh'].values[0]

			house_rf = df_filled['House ID'] == missing_house
			interval_rf = df_filled['Interval'] == missing_interval
			df_filled['kWh'][house_rf&interval_rf] = house_half_hour_average

		except:
			print(row)
			break

	return df_filled


def house_agg_and_split(df, has_label=True):
	
	half_hours = np.arange(1,49)

	# calculate mean and standard deviation of each half hour
	df_described = df.groupby(['House ID', 'Half Hour']
		).agg({'kWh':['mean',np.std]}
		).pivot_table(values='kWh',index=['House ID'], columns='Half Hour'
		).reset_index()

	mean_cols = ['u_{}'.format(n) for n in half_hours]
	sd_cols = ['s_{}'.format(n) for n in half_hours]
	hds_cols = ['House ID'] + mean_cols + sd_cols
	df_described.columns = hds_cols

	# calculate average kWh as percent of day
	pct_cols = ['p_{}'.format(n) for n in half_hours]
	df_described[pct_cols] = df_described[mean_cols].div(df_described[mean_cols].sum(axis=1),axis=0)

	if has_label:
		# create label at the house level and merge with house dataset to keep order
		df_labels = df.groupby(['House ID']).agg({'Label':sum}).reset_index()
		df_labels['House Label'] = [1 if x > 0 else 0 for x in df_labels['Label']]
		df_labels.drop(['Label'], axis=1, inplace=True)
		h_Xy = df_described.merge(df_labels, how='left', on='House ID')

		# split X and y
		h_X = h_Xy.copy()
		h_y = h_X['House Label']
		h_X.drop(['House Label'], axis=1, inplace=True)

		return h_X, h_y

	else:

		h_X = df_described.copy()
		
		return h_X



def cluster_retrieve_best_k(h_X, h_y, k_min=2, k_max=10):

	max_k_percent = 0

	best_k = None
	best_k_col_name = None
	h_X_with_cluster = None
	cluster_centers_df = None

	k_values = np.arange(k_min, k_max+1)

	pct_cols = ['p_{}'.format(n) for n in np.arange(1,49)]

	for k in k_values:

		h_X_with_k = h_X.copy()
		
		# assign cluster
		kmeans_model = KMeans(n_clusters=k, random_state=42).fit(h_X_with_k[pct_cols].as_matrix())
		k_col_name = "k_{}".format(k)
		h_X_with_k[k_col_name] = kmeans_model.labels_

		# calculate ratio between clusters with max and min amount of EV labeled houses 
		count_test = h_X_with_k.copy()
		count_test['Label'] = h_y
		count_test_percent = count_test.groupby(k_col_name)['Label'].sum()/count_test.groupby(k_col_name)['Label'].count()
		k_percent = (max(count_test_percent) / min(count_test_percent))
	    
	 	# save k that maximizes ratio
		if k_percent > max_k_percent:
			max_k_percent = k_percent
			best_k = k
			best_k_col_name = k_col_name
			h_X_with_cluster = h_X_with_k
			cluster_centers_df = kmeans_model.cluster_centers_

	# describe results
	print("Best clustering was when k = {}, with a ratio of {:.3f}".format(best_k, max_k_percent))
	pd.DataFrame(cluster_centers_df).T.plot()

	# clean up centers_df
	cluster_centers_df = pd.DataFrame(cluster_centers_df).T.reset_index()
	cluster_centers_df.columns = ['Half Hour']+["c{}".format(k) for k in np.arange(1,best_k+1).tolist()]
	cluster_centers_df['Half Hour'] = cluster_centers_df['Half Hour']+1

	return h_X_with_cluster, cluster_centers_df, best_k_col_name


def transform_house_subset_to_time(h_X, h_y, all_processed_df, has_label=True):

	house_ids = h_X['House ID'][h_y>0].values # TODO: change to complete subset house IDs and EV house IDs
	keep_records = all_processed_df['House ID'].isin(house_ids)
	t_X = all_processed_df[keep_records]

	if has_label:

		t_y = t_X['Label']
		t_X.drop(['Label'], axis=1, inplace=True)
		
		return t_X, t_y

	else:

		return t_X


def augment_time_data(t_X, h_X, centers_df, k_col='k_9', diff_days=7):

	# day aggregation
	day_totals_df = t_X.groupby(['House ID','Day']).agg({'kWh':sum}).reset_index()
	day_totals_df.columns = ['House ID', 'Day', 'Day_kWh'] 

	# join
	t_X_aug = t_X.merge(day_totals_df, how='left', on=['House ID', 'Day'])
	t_X_aug = t_X_aug.merge(h_X[['House ID', k_col]], how='left', on='House ID')
	t_X_aug = t_X_aug.merge(centers_df, how='left', on='Half Hour')

	# scale up centers
	c_names = centers_df.columns.values.tolist()
	c_names.pop(0)

	t_X_aug[c_names] = t_X_aug[c_names].multiply(t_X_aug['Day_kWh'], axis='index')

	# calculate day difference
	for d in np.arange(1,diff_days+1):
		day_diff_col_name = "{}d_diff".format(d)
		t_X_aug[day_diff_col_name] = t_X_aug.groupby(['House ID','Half Hour'])['kWh'].diff(periods=d)

	# fill na to run model
	t_X_aug.fillna(value=0, inplace=True)

	return t_X_aug


def show_baseline_f1_scores(y):

	zero_zero = np.tile([0,0],math.floor(len(y)/2))
	if len(zero_zero) < len(y):
		zero_zero.append(0)
	
	zero_one = np.tile([0,1],math.floor(len(y)/2))
	if len(zero_zero) < len(y):
		zero_zero.append(0)
	
	one_one = np.tile([1,1],math.floor(len(y)/2))
	if len(zero_zero) < len(y):
		zero_zero.append(1)

	print ("All 0 F-score: {0:.3f}".format(f1_score(y, zero_zero)))
	print ("Alternating 0,1 F-score: {0:.3f}".format(f1_score(y, zero_one)))
	print ("All 1 F-score: {0:.3f}".format(f1_score(y, one_one)))


def format_ev_predictions(y, X, first_int=1, final_int=2880):

	preds_long = X[['House ID', 'Interval']].copy()
	preds_long['Label'] = y

	preds = preds_long.pivot(index='House ID',columns='Interval')

	preds.columns = preds.columns.get_level_values(0)
	preds.columns = ["Interval_{}".format(i) for i in np.arange(first_int, final_int+1).tolist()]

	preds = preds.reset_index()

	return preds


def complete_time_predictions(raw, preds):

	h_order_df = raw['House ID'].to_frame()

	complete = h_order_df.merge(preds, how='left', on='House ID')

	complete = complete.fillna(0)

	int_cols = complete.columns.values.tolist()[1:]

	complete[int_cols] = complete[int_cols].astype(int)

	return complete
