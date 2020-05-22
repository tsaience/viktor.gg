
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


def process_champ_name(champ_name):
	split_by_space = "".join(champ_name.split(" "))
	split_by_apostrophe = "".join(split_by_space.split("'"))
	split_by_amp = "".join(split_by_apostrophe.split("&"))
	lowercase = split_by_amp.lower()
	return lowercase


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




champ_chars = pd.read_csv('data/champ_chars.csv')
champ_chars.name = [process_champ_name(champ_chars.name[i]) for i in range(len(champ_chars.index.values))]
champ_chars.set_index('name', inplace = True)


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
