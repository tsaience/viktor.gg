import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import cassiopeia as cass
import json 
import requests  
import urllib.request
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# league: 'LEC', 'LCS', 'LPL', 'LCK'
# year: self-explanatory
# split: 'Spring', 'Summer'

def construct_url_for_split(league, split, year):
	url = "https://lol.gamepedia.com/" + league + "/" + str(year) + "_Season/" + split + "_Season/Picks_and_Bans"
	return url






################## LOADING DATA ##################


url = "http://ddragon.leagueoflegends.com/cdn/10.10.3216176/data/en_US/champion.json" 
r = requests.get(url=url) 
json_file = r.json()
data = json_file['data']

champ_names = list(data.keys())
name = champ_names[0]
cols = list(['name'])
cols.extend(list(data[name]['stats'].keys()))

df = pd.DataFrame(columns = cols)
for name in champ_names:
	dict_temp = dict()
	dict_temp['name'] = data[name]['id']
	dict_temp.update(data[name]['stats'])
	df_temp = pd.DataFrame(dict_temp, index = [0])
	df = df.append(df_temp, ignore_index = True)



champ_chars = pd.read_csv('data/champ_chars.csv')
champ_chars.name = [process_champ_name(champ_chars.name[i]) for i in range(len(champ_chars.index.values))]
champ_chars.set_index('name', inplace = True)


df.name = [process_champ_name(df.name[i]) for i in range(len(df.index.values))]
df.set_index('name', inplace = True)




# Merge data from Riot API (Armor / AD / AS etc.) with binary champ characteristics

df_full = pd.merge(df,champ_chars, on = 'name')

# Alternatively use only our champ characteristics
df_full = champ_chars


print(df_full.head())




################## Getting Game Data ##################

# region - split - years of interest
params = [('LPL','Spring',2020),('LCK','Spring',2020),('LEC','Spring',2020),('LCS','Spring',2020)]#,
#		('LPL','Spring',2019),('LCK','Spring',2019),('LEC','Spring',2019),('LCS','Spring',2019),
#		('LPL','Summer',2019),('LCK','Summer',2019),('LEC','Summer',2019),('LCS','Summer',2019),
#		('LPL','Spring',2018),('LCK','Spring',2018)]



# Make columns for df_games which will have game data
chars_cols = df_full.columns.values
sides = ['blue','red']

# From features
cols_1 = ['blue_'+name for name in df_full]
cols_2 = ['red_'+name for name in df_full]

# Adding indicator for winning team
cols = ['blue_win']+cols_1+cols_2

df_games = pd.DataFrame(columns = cols)



# Loop over splits
for param in params:
	url = construct_url_for_split(*param)
	table_rows = extract_table_rows_from_url(url)

	# Loop games within each split
	for row in table_rows:
		game = extract_game_from_row(row, verbose = False) 

		# Subset df_full (which contains champ chars. and more) to find blue team's champs
		blue_champs = list(game.blue_champs)
		blue_ind = np.in1d(df_full.index.values, blue_champs)
		blue_mat = df_full.iloc[blue_ind]

		# Same for red side
		red_champs = list(game.red_champs)
		red_ind = np.in1d(df_full.index.values, red_champs)
		red_mat = df_full.iloc[red_ind]

		# Take sums along columns to create game featureset
		temp_b = blue_mat.sum(axis = 0)
		temp_r = red_mat.sum(axis = 0)


		# Game data: [blue won, blue features, red features]
		df_temp = pd.DataFrame([[int(game.blue_win)] + temp_b.values.tolist() + temp_r.values.tolist()], columns = cols)

		df_games = df_games.append(df_temp, ignore_index = True)


print(df_games.head())

df_games.to_csv('data/game_stats.csv', index = False)





df_games = pd.read_csv('data/game_stats.csv')

# Perform PCA
n_comp = 1

pca = PCA(n_comp)
scaler = StandardScaler()

df_fitted = pd.DataFrame(pca.fit_transform(scaler.fit_transform(df_games.iloc[:,1:])),
						 index = df_games.index, columns = ['PC'+str(i) for i in range(1,n_comp+1)])

print(df_games.iloc[:,1:].head())
print(df_fitted.head())



# Testing Logistic Regression
X_pc = df_fitted.to_numpy()

# Comment out one of these to use full data or PC's
X = df_games.iloc[:,1:].to_numpy()
X = X_pc

y = df_games.blue_win.to_numpy()


X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42)


lr_model = LogisticRegression()

lr_model.fit(X_train, y_train)

example_predict = lr_model.predict(X_test)

print(lr_model.predict_proba(X_test))
print(lr_model.score(X_test,y_test))
print(y.mean())




