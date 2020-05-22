

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
import seaborn as sns



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


df_full = pd.merge(df,champ_chars, on = 'name')


df_full = champ_chars


def process_champ_name(champ_name):
	split_by_space = "".join(champ_name.split(" "))
	split_by_apostrophe = "".join(split_by_space.split("'"))
	split_by_amp = "".join(split_by_apostrophe.split("&"))
	lowercase = split_by_amp.lower()
	return lowercase

df_full.name = [process_champ_name(df_full.name[i]) for i in range(len(df_full.index.values))]


df_full.set_index('name', inplace = True)

#print(df_full)



# Check Variance
for col in df_full.columns:
	if np.var(df_full.loc[:,col]) < 0.01:
		df_full.drop(col, axis = 1, inplace = True)
		print(f'Dropped {col}')





# Perform PCA
n_comp = 5

pca = PCA(n_comp)
scaler = StandardScaler()

df_fitted = pd.DataFrame(pca.fit_transform(scaler.fit_transform(df_full)),
						 index = df_full.index, columns = ['PC'+str(i) for i in range(1,n_comp+1)])



for col in df_fitted.columns:
	temp = df_fitted.loc[:,col] 

	print(f'\n{col}: Top Champions')
	print(temp.sort_values(ascending = False).iloc[0:10])


	print(f'\n{col}: Lowest Champions')
	print(temp.sort_values(ascending = True).iloc[0:10])





def eff_rank(A):
	AtA = np.dot(np.transpose(A),A)

	return np.trace(AtA)/np.linalg.norm(A, ord = 2)



#data structure

class LeagueGame:
	def __init__(self, blue_champs, red_champs, blue_win, patch):
		self.blue_champs = blue_champs
		self.red_champs = red_champs
		self.blue_win = blue_win
		self.patch = patch


# league: 'LEC', 'LCS', 'LPL', 'LCK'
# year: self-explanatory
# split: 'Spring', 'Summer'

def construct_url_for_split(league, split, year):
	url = "https://lol.gamepedia.com/" + league + "/" + str(year) + "_Season/" + split + "_Season/Picks_and_Bans"
	return url


def extract_table_rows_from_url(url):
	fp = urllib.request.urlopen(url)
	page_bytes = fp.read()
	page_html = page_bytes.decode("utf8")
	fp.close()
	soup = BeautifulSoup(page_html, 'html.parser')
	draft_table = soup.find_all('table', id="pbh-table")[0]
	table_rows = draft_table.find_all('tr')[2:]
	return table_rows




def extract_game_from_row(row_of_interest, verbose=True):

	table_columns = row_of_interest.find_all('td')
	blue, red = table_columns[1], table_columns[2]

	blue_win = False
	if blue.get("class") != None and blue.get("class")[0] == "pbh-winner":
		blue_win = True

	patch = table_columns[4].get_text()

	champs_columns = table_columns[11:15] + table_columns[19:22]

	# bp1, rp1-2, bp2-3, rp3, rp4, bp4-5, rp5

	non_champ_classes = {'pbh-blue', 'pbh-cell', 'pbh-red', 'pbh-divider'}

	blue_champs = set()
	red_champs = set()

	for i in range(len(champs_columns)):
		column = champs_columns[i]
		classes = column.get("class")
		for column_class in classes:
			if column_class not in non_champ_classes:
				champ = column_class.split("-")[1]
				if i in {0, 2, 5}:
					blue_champs.add(champ)
				else:
					red_champs.add(champ)

	game = LeagueGame(blue_champs, red_champs, blue_win, patch)

	if verbose:
		print("game blue champs: ", game.blue_champs)
		print("game red champs: ", game.red_champs)
		print("game blue win: ", game.blue_win)
		print("game patch: ", game.patch)
		print()

	return game












'''

df_score = pd.DataFrame(columns = ['score', 'win'])



# scrap every row
url = construct_url_for_split("LCK", "Summer", 2018)
table_rows = extract_table_rows_from_url(url)
for row in table_rows:
	game = extract_game_from_row(row, verbose = False) # set this to false if you dont want to see prints


	## THIS IS WHERE WE DEFINE A SCORE FUNCTION

	blue_champs = list(game.blue_champs)

	blue_ind = np.in1d(df_fitted.index.values, blue_champs)
	blue_mat = df_fitted.iloc[blue_ind]

	blue_score = eff_rank(blue_mat)

	df_temp = pd.DataFrame({'score': blue_score, 'win': game.blue_win}, index = [0])

	df_score = df_score.append(df_temp, ignore_index = True)


print(df_score)




n_buckets = 20

ticks = np.linspace(min(df_score.score), max(df_score.score), n_buckets)

x = np.zeros(n_buckets-1)
y = np.zeros(n_buckets-1)

for i in range(len(ticks)-1):
	x[i] = 0.5*(ticks[i]+ticks[i+1])

	ind = (ticks[i] <= df_score.score) & (df_score.score <= ticks[i+1])

	y[i] = df_score.win.loc[ind].mean()

print(x)
print(y)


f,ax = plt.subplots(nrows = 1, ncols = 1, figsize = (7,7))

sns.pointplot(x = x, y = y)

plt.savefig('temp.pdf')























red_champs = ['ornn', 'leesin', 'anivia', 'kaisa', 'nautilus']

blue_champs = ['jayce', 'gragas', 'zoe', 'corki', 'janna']


red_ind = np.in1d(df_fitted.index.values, red_champs)
red_mat = df_fitted.iloc[red_ind]

print(red_mat)

print(eff_rank(red_mat))



blue_ind = np.in1d(df_fitted.index.values, blue_champs)
blue_mat = df_fitted.iloc[blue_ind]

print(blue_mat)

print(eff_rank(blue_mat))





'''

