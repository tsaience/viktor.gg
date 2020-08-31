

import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
#import cassiopeia as cass
import json 
import requests  
import urllib.request
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
#import seaborn as sns





def process_names(champ_name):
	split_by_space = "".join(champ_name.split(" "))
	split_by_apostrophe = "".join(split_by_space.split("'"))
	split_by_amp = "".join(split_by_apostrophe.split("&"))
	lowercase = split_by_amp.lower()
	return lowercase



def load_data():
    
    url = "http://ddragon.leagueoflegends.com/cdn/10.16.1/data/en_US/champion.json" 
    r = requests.get(url=url) 
    
    json_file = r.json()
    
    data = json_file['data']
    
    champ_names = list(data.keys())
    name = champ_names[0]
    cols = list(['name'])
    cols.extend(list(data[name]['stats'].keys()))
    cols.extend(list(data[name]['info'].keys()))
    cols.extend(['tags'])
    
    df = pd.DataFrame(columns = cols)

    
    # Input champ data 
    for name in champ_names:
        dict_temp = dict()
        dict_temp['name'] = data[name]['name']
        
        dict_temp.update(data[name]['stats'])
        dict_temp.update(data[name]['info'])
        dict_temp['tags'] = [data[name]['tags']]
                
        df_temp = pd.DataFrame(dict_temp, index = [0])
        
        df = df.append(df_temp, ignore_index = True)
            
        
    # Process champ names for consistency
    df.name = [process_names(df.name[i]) for i in range(len(df.index.values))]
    
    df.set_index('name', inplace = True)
    
    
    # One-hot encoding of riot's tags   
    mlb = MultiLabelBinarizer()
    
    df_onehot = pd.DataFrame(mlb.fit_transform(df.loc[:,'tags']), columns = mlb.classes_, index = df.index)
    
    df = df.drop('tags', axis = 1).merge(df_onehot, left_index = True, right_index = True)
    
    df.columns = [process_names(item) for item in df.columns.values]
    
    
    return (json_file, df)



def pca_analysis(df, n_comp, n_champs_print = 10, plot = False, show = False, scale = True):
    
    pca = PCA(n_comp)

    if scale:
        scaler = StandardScaler()
        temp = pca.fit_transform(scaler.fit_transform(df))

    else:
        temp = pca.fit_transform(df)
    
    
    # Data of output matrix
    df_fitted = pd.DataFrame(temp, index = df.index,
                             columns = ['PC'+str(i) for i in range(1,n_comp+1)])

    # Show top/bot n_champs_print champions for each PC
    if show:
        for col in df_fitted.columns:
        	temp = df_fitted.loc[:,col] 
        
        	print(f'\n{col}: High + Low Champions')
        	print(temp.sort_values(ascending = False).iloc[0:n_champs_print].reset_index().values)
        	print('...')
        	print(temp.sort_values(ascending = True).iloc[0:n_champs_print].sort_values(ascending = False).reset_index().values)
    
    # Plot 1st two PC's
    if plot:
        x = df_fitted.PC1
        y = df_fitted.PC2
        
        fig, ax = plt.subplots(figsize = (7,7))
        ax.scatter(x, y)
        for i, txt in enumerate(df_fitted.index.values):
            ax.annotate(txt, (x[i], y[i]))
        plt.show()
        
    return df_fitted
    


def cluster_analysis(df, n_clusters = 3, plot = False, show = False, n_init = 50):
    
    kmeans = KMeans(n_clusters, n_init = n_init, random_state = 0)
    
    kmeans.fit(df)
    
    df_out = pd.DataFrame(kmeans.labels_, index = df.index, columns = ['labels'])
    
    
    if show:
        print(df_out.sort_values('labels'))

    if plot:
        x = df.PC1
        y = df.PC2
        
        fig, ax = plt.subplots(figsize = (7,7))
        ax.scatter(x, y, kmeans.labels_)
        for i, txt in enumerate(df_fitted.index.values):
            ax.annotate(txt, (x[i], y[i]))
        plt.show()
    
    
    return df_out
    
    
    


###################################################################################################
###################################################################################################
###################################################################################################
    



(json_file, df) = load_data()


tops = ['camille','shen','jax','renekton','darius','garen','fiora','mordekaiser','irelia','riven','malphite','sett','akali','wukong','aatrox','yone','maokai','jayce','vladimir','ornn','tryndamere','volibear','sylas','gangplank','teemo','nasus','urgot','gnar','poppy','kayle','yasuo','lucian','sion','singed','kled','kennen','illaoi','chogath','yorick','rengar','hecarim','lillia','vayne','drmundo','quinn','rumble']
jungs = ['leesin','kayn','khazix','ekko','graves','hecarim','evelynn','elise','nunuwillump','zac','nidalee','volibear','masteryi','lillia','shaco','fiddlesticks','karthus','vi','kindred','olaf','reksai','sylas','warwick','rengar','shyvana','jarvaniv','sett','nocturne','amumu','gragas','rammus','sejuani','ivern','udyr','skarner','jax','xinzhao','trundle']
mids = ['yone', 'zed', 'yasuo', 'akali', 'sylas', 'ahri', 'orianna', 'kassadin', 'katarina', 'zoe', 'fizz', 'galio', 'vladimir', 'ekko', 'leblanc', 'cassiopeia', 'talon', 'diana', 'twistedfate', 'lux', 'irelia', 'syndra', 'veigar', 'lucian', 'qiyana', 'annie', 'malzahar', 'pantheon', 'viktor', 'lissandra', 'azir', 'velkoz', 'ryze', 'ziggs', 'xerath', 'neeko', 'pyke', 'anivia', 'swain', 'renekton', 'sett', 'corki']
adcs = ['caitlyn', 'ezreal', 'ashe', 'jhin', 'lucian', 'kaisa', 'vayne', 'jinx', 'tristana', 'aphelios', 'draven', 'missfortune', 'senna', 'twitch', 'xayah', 'kalista', 'yasuo', 'sivir', 'kogmaw', 'lux', 'varus']
sups = ['thresh', 'lulu', 'lux', 'morgana', 'senna', 'yuumi', 'nautilus', 'karma', 'blitzcrank', 'bard', 'leona', 'pyke', 'soraka', 'sona', 'nami', 'janna', 'pantheon', 'rakan', 'swain', 'zyra', 'alistar', 'sett', 'zilean', 'brand', 'braum', 'xerath', 'taric', 'velkoz', 'shaco', 'maokai', 'galio']


roles = [tops, jungs, mids, adcs, sups]
role_names = ['Top','Jungle','Mid','ADC','Support']


n_comp = 4
n_champs_print = 5
plot = True
show = False
scale = True

n_clusters = 3
n_init = 50

for i,role in enumerate(roles):
    
    print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(role_names[i])
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    
    
    df_temp = df.loc[role,:].dropna(axis = 0)
    
    df_fitted = pca_analysis(df_temp, n_comp, n_champs_print, plot, show, scale)
    
    df_clusters = cluster_analysis(df_fitted, n_clusters, plot, show, n_init)










'''
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







# scrap every row
params = [('LPL','Spring',2020)]#,('LCK','Spring',2020),('LEC','Spring',2020),('LCS','Spring',2020),
#		('LPL','Spring',2019),('LCK','Spring',2019),('LEC','Spring',2019),('LCS','Spring',2019),
#		('LPL','Summer',2019),('LCK','Summer',2019),('LEC','Summer',2019),('LCS','Summer',2019),
#		('LPL','Spring',2018),('LCK','Spring',2018)]



chars_cols = champ_chars.columns.values
sides = ['blue','red']

cols_1 = ['blue_'+name for name in chars_cols]
cols_2 = ['red_'+name for name in chars_cols]

cols = ['blue_win']+cols_1+cols_2


df_games = pd.DataFrame(columns = cols)


for param in params:
	url = construct_url_for_split(*param)
	table_rows = extract_table_rows_from_url(url)
	for row in table_rows:
		game = extract_game_from_row(row, verbose = False) 


		blue_champs = list(game.blue_champs)

		blue_ind = np.in1d(champ_chars.index.values, blue_champs)
		blue_mat = champ_chars.iloc[blue_ind]

		red_champs = list(game.red_champs)

		red_ind = np.in1d(champ_chars.index.values, red_champs)
		red_mat = champ_chars.iloc[red_ind]

		#print(blue_mat)
		#print(red_mat)

		temp_b = blue_mat.sum(axis = 0)
		temp_r = red_mat.sum(axis = 0)


		df_temp = pd.DataFrame([[int(game.blue_win)] + temp_b.values.tolist() + temp_r.values.tolist()], columns = cols)

		df_games = df_games.append(df_temp, ignore_index = True)


print(df_games)

'''










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

