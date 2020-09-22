

import numpy as np
import pandas as pd
import random
import json 
import requests  
import urllib.request

import matplotlib.pyplot as plt

from itertools import product

from bs4 import BeautifulSoup

from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram


# Our Code
from models import LeagueGame





def process_names(champ_name):
    split_by_space = "".join(champ_name.split(" "))
    split_by_apostrophe = "".join(split_by_space.split("'"))
    split_by_amp = "".join(split_by_apostrophe.split("&"))
    split_by_period = "".join(split_by_amp.split("."))
    split_by_period = "".join(split_by_amp.split("."))    
    lowercase = split_by_period.lower()
    return lowercase


def load_data(patch):
    
    url = "http://ddragon.leagueoflegends.com/cdn/"+patch+"/data/en_US/champion.json" 
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
    
    return (json_file, df, list(data.keys()))


def get_core_items(champs_all, patch, champ_type = None):
    
    items_all = dict()
    
    for champ in champs_all:
        
        url = "http://ddragon.leagueoflegends.com/cdn/"+patch+"/data/en_US/champion/"+champ+".json"
        r = requests.get(url=url) 
        
        json_file = r.json()
        
        data_temp = json_file['data'][champ]
            
        # Potato-coding edge cases
        if champ == 'Fiddlesticks':
            champ = 'Fiddle'
        if champ =='MonkeyKing':
            champ = 'Wukong'
        
        # Search for Map
        # DOES NOT DEFAULT TO ROLE-SPECIFIC ITEMIZATION
        ids = np.where([process_names(temp['title'])==process_names(champ+'SR') for temp in data_temp['recommended']])[0]
        items_temp = data_temp['recommended'][ids[0]]['blocks']
        
        # Search for Core Items
        if champ_type == 'jungle':
            ids = np.where([temp['type']=='essentialjungle' for temp in items_temp])[0]
            
            if len(ids)==0:
                ids = np.where([temp['type']=='essential' for temp in items_temp])[0]        
            
        else:
            ids = np.where([temp['type']=='essential' for temp in items_temp])[0]
    
        # More potato-coding edge cases
        # Only Qiyana has no Essential Items - use 'early' instead
        if len(ids)==0:
            if champ_type == 'jungle':
                ids = np.where([temp['type']=='earlyjungle' for temp in items_temp])[0]
                
                if len(ids)==0:
                    ids = np.where([temp['type']=='early' for temp in items_temp])[0]        
                
            else:
                ids = np.where([temp['type']=='early' for temp in items_temp])[0]
        
        # Save to Output
        items_champ = [item['id'] for item in items_temp[ids[0]]['items']]
        items_all[process_names(data_temp['name'])] = [items_champ]
    
    item_df = pd.DataFrame(items_all, index = ['Core_Items']).transpose()
        
    mlb = MultiLabelBinarizer()
    item_df = pd.DataFrame(mlb.fit_transform(item_df['Core_Items']), columns = mlb.classes_, index = item_df.index)
    
    return item_df
    

def pca_analysis(df, n_comp, n_champs_print = 10, plot = False, show = False, scale = True, scale_cols = None):
    
    pca = PCA(n_comp)

    if scale_cols is None:
        scale_cols = df.columns.values

    if scale:
        scaler = StandardScaler()
        df_temp = pd.DataFrame(scaler.fit_transform(df.loc[:,scale_cols]), index = df.index.values, columns = df.loc[:,scale_cols].columns.values)
        
        df = df_temp.merge(df.drop(scale_cols, axis = 1), how = 'inner', left_index = True, right_index = True)
        


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
    

def fit_clusters(df, figname, n_clusters = 3, plot = False, show = False, n_init = 50, clstr = 'kmeans'):
    
    colors = ['red','blue','green','orange','yellow','brown','gray']
    
    if clstr == 'kmeans':
        model = KMeans(n_clusters, n_init = n_init, random_state = 0)
        
        model.fit(df)
        
        c = [colors[i] for i in model.labels_]
        
        df_out = pd.DataFrame(model.labels_, index = df.index, columns = ['labels'])
        
        if show:
            print(df_out.sort_values('labels'))
    
        if plot:
            x = df.PC1
            y = df.PC2
            
            fig, ax = plt.subplots(figsize = (7,7))
            ax.scatter(x, y, c = c)
            for i, txt in enumerate(df.index.values):
                ax.annotate(txt, (x[i], y[i]))
            plt.title(figname)
            plt.savefig('fig/'+figname+'.pdf')
            plt.show()
        
        return (model,df_out)
    
    elif clstr == 'agg':
        
        if n_clusters is None:
            model = AgglomerativeClustering(distance_threshold = 0, n_clusters = None, linkage = 'ward')

            model.fit(df)
            
            if plot:
                plot_dendrogram(model, figname, labels = df.index.values)
            
            return (model, df)
        
        else:
            model = AgglomerativeClustering(distance_threshold = None, n_clusters = n_clusters, linkage = 'ward')

            model.fit(df)
            
            c = [colors[i] for i in model.labels_]

            df_out = pd.DataFrame(model.labels_, index = df.index, columns = ['labels'])

            if plot:
                x = df.PC1
                y = df.PC2
                
                fig, ax = plt.subplots(figsize = (7,7))
                ax.scatter(x, y, c = c)
                for i, txt in enumerate(df.index.values):
                    ax.annotate(txt, (x[i], y[i]))
                plt.title(figname)
                plt.savefig('fig/'+figname+'.pdf')
                plt.show()
            
            return (model, df_out)
        
    else:
        Exception('Unrecognized clustering type')
    
    
def plot_dendrogram(model, figname, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    f,ax = plt.subplots(figsize = (10,7))

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    
    dendrogram(linkage_matrix, **kwargs)
    plt.xticks(rotation=90)
    plt.title(figname)
    plt.savefig('fig/'+figname+'.pdf')
    plt.show()




def run_clustering_analysis(df, item_df_jung, item_df_lane, roles, role_names, 
                            n_clusters = [2,2,2,2,2], clstr = 'kmeans', n_comp = 6, 
                            n_champs_print = 5, plot = True, show = False, 
                            scale = True, scale_cols = None, n_init = 50):
    
    cluster_dicts = dict()
    
    for i,role in enumerate(roles):
    
        print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(role_names[i])
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        
        if role_names[i]=='Jungle':
            df_items = item_df_jung.loc[role,:].dropna(axis = 0)
        else:
            df_items = item_df_lane.loc[role,:].dropna(axis = 0)
        
        df_champ = df.loc[role,:].dropna(axis = 0)
        
        df_temp = df_champ#.merge(df_items, how = 'outer', left_index = True, right_index = True)
        
        df_fitted = pca_analysis(df_temp, n_comp, n_champs_print, False, show, scale, scale_cols)
        
        # do clustering
        (model,df_clusters) = fit_clusters(df_fitted, role_names[i], n_clusters[i], plot, show, n_init, clstr = clstr)
    
        cluster_dicts[role_names[i]] = df_clusters.to_dict()['labels']
    
    # Save clustering dict
    with open('clusters.json','w') as fp:
        json.dump(cluster_dicts, fp)
        
    return cluster_dicts



###################################################################################################
###################################################################################################
###################################################################################################
    


# Get all Champ and Item Data
(json_file, df, champ_list) = load_data(patch = "10.19.1")
item_df_jung = get_core_items(champ_list, patch = "10.19.1", champ_type = 'jungle')
item_df_lane = get_core_items(champ_list, patch = "10.19.1", champ_type = None)



# Manual Input Here
tops = ['camille','shen','jax','renekton','darius','garen','fiora','mordekaiser','irelia','riven','malphite','sett','akali','wukong','aatrox','yone','maokai','jayce','vladimir','ornn','tryndamere','volibear','sylas','gangplank','teemo','nasus','urgot','gnar','poppy','kayle','yasuo','lucian','sion','singed','kled','kennen','illaoi','chogath','yorick','rengar','hecarim','lillia','vayne','drmundo','quinn','rumble']
jungs = ['leesin','kayn','khazix','ekko','graves','hecarim','evelynn','elise','nunuwillump','zac','nidalee','volibear','masteryi','lillia','shaco','fiddlesticks','karthus','vi','kindred','olaf','reksai','sylas','warwick','rengar','shyvana','jarvaniv','sett','nocturne','amumu','gragas','rammus','sejuani','ivern','udyr','skarner','jax','xinzhao','trundle']
mids = ['yone', 'zed', 'yasuo', 'akali', 'sylas', 'ahri', 'orianna', 'kassadin', 'katarina', 'zoe', 'fizz', 'galio', 'vladimir', 'ekko', 'leblanc', 'cassiopeia', 'talon', 'diana', 'twistedfate', 'lux', 'irelia', 'syndra', 'veigar', 'lucian', 'qiyana', 'annie', 'malzahar', 'pantheon', 'viktor', 'lissandra', 'azir', 'velkoz', 'ryze', 'ziggs', 'xerath', 'neeko', 'pyke', 'anivia', 'swain', 'renekton', 'sett', 'corki']
#adcs = ['caitlyn', 'ezreal', 'ashe', 'jhin', 'lucian', 'kaisa', 'vayne', 'jinx', 'tristana', 'aphelios', 'draven', 'missfortune', 'senna', 'twitch', 'xayah', 'kalista', 'yasuo', 'sivir', 'kogmaw', 'lux', 'varus']
adcs = ['caitlyn', 'ezreal', 'ashe', 'jhin', 'lucian', 'kaisa', 'vayne', 'jinx', 'tristana', 'aphelios', 'draven', 'missfortune', 'senna', 'twitch', 'xayah', 'kalista', 'sivir', 'kogmaw', 'varus']
sups = ['thresh', 'lulu', 'lux', 'morgana', 'senna', 'yuumi', 'nautilus', 'karma', 'blitzcrank', 'bard', 'leona', 'pyke', 'soraka', 'sona', 'nami', 'janna', 'pantheon', 'rakan', 'swain', 'zyra', 'alistar', 'sett', 'zilean', 'brand', 'braum', 'xerath', 'taric', 'velkoz', 'shaco', 'maokai', 'galio']



# Parameters for Clustering
roles = [tops, jungs, mids, adcs, sups]
role_names = ['Top','Jungle','Mid','ADC','Support']


n_comp = 10
n_champs_print = 5
plot = True
show = False
scale = True
clstr = 'agg'
n_clusters = [3,2,2,3,2]
n_init = 50
scale_cols = df.columns.values[:20] #all base and scaling stats


# Run Clustering Analysis
cluster_dicts = run_clustering_analysis(df, item_df_jung, item_df_lane, roles, role_names, 
                            n_clusters = n_clusters, clstr = clstr, n_comp = n_comp, 
                            n_champs_print = n_champs_print, plot = plot, show = show,
                            scale = scale, scale_cols = scale_cols, n_init = n_init)








###################################################################################################
###################################################################################################
###################################################################################################


clusters = [np.unique(list(cluster_dicts[name].values())).tolist() for name in role_names]

team_clusters = list(product(*clusters))

index = ["".join([str(idx) for idx in curr]) for curr in team_clusters]

team_wins = pd.DataFrame(np.zeros((len(index),len(index))), index = index, columns = index)
team_games = pd.DataFrame(np.zeros((len(index),len(index))), index = index, columns = index)




# Loop over Games
game = LeagueGame(['ornn','leesin','syndra','ashe','nautilus'],['gangplank','nunuwillump','ekko','caitlyn','lux'],True, "10.19.1")

blueid = "".join([str(cluster_dicts[role_names[i]][champ]) for i,champ in enumerate(game.blue_champs)])
redid = "".join([str(cluster_dicts[role_names[i]][champ]) for i,champ in enumerate(game.red_champs)])

team_games.loc[blueid,redid] = team_games.loc[blueid,redid] + 1 
if game.blue_win:
    team_wins.loc[blueid,redid] = team_wins.loc[blueid,redid] + 1 
        
# End Loop


team_winrate = team_wins/team_games










'''

