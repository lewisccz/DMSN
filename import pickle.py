import pickle
import networkx as nx 
import matplotlib.pyplot as plt
import csv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy import SpotifyException
import time
import numpy as np

pickle_file_path = 'small_graph.pickle'

# Open the .pickle file and load its contents
with open(pickle_file_path, 'rb') as file:
    G = pickle.load(file)

# Load the CSV file
song_info = {}
with open('song_data_smaller.csv', 'r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        song_id = row['song_id']
        song_info[song_id] = row


client_id = '4be71c669546403dbc75319d0ea0601a'
client_secret = '2b7db59168d64b1887a9f9a8af58553f'

client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_track_data(song_ids):
    retries = 0
    max_retries = 3
    wait_time = 1
    all_track_data = []

    while retries < max_retries:
        try:
            audio_features_list = sp.audio_features(song_ids)
            track_details_list = sp.tracks(song_ids)['tracks']
            
            for audio_features, track_details in zip(audio_features_list, track_details_list):
                if track_details:
                    # Assuming the first artist is the primary one
                    artist_uri = track_details['artists'][0]['uri']
                    artist_details = sp.artist(artist_uri)
                    artist_popularity = artist_details['popularity']
                else:
                    artist_popularity = None

                track_data = {
                    'audio_features': audio_features,
                    'popularity': track_details['popularity'] if track_details else None,
                    'artist_popularity': artist_popularity
                }
                all_track_data.append(track_data)
            return all_track_data
        except SpotifyException as e:
            if e.http_status == 429 or 'max retries' in str(e):
                print(f"Rate limited. Waiting for {wait_time} seconds.")
                time.sleep(wait_time)
                wait_time *= 2
                retries += 1
            else:
                raise
    return None


batch_size = 50
song_ids = list(song_info.keys())  # Get a list of all song IDs from the dictionary

for i in range(0, len(song_ids), batch_size):
    batch_song_ids = song_ids[i:i+batch_size]  # Collect a batch of song IDs
    tracks_data = get_track_data(batch_song_ids)  # Fetch track data for the batch

    if tracks_data:
        for track_data, song_id in zip(tracks_data, batch_song_ids):
            if track_data and track_data['audio_features']:
                # Update each song's information in `song_info` dictionary
                song_info[song_id]['energy'] = track_data['audio_features']['energy']
                song_info[song_id]['valence'] = track_data['audio_features']['valence']
                song_info[song_id]['danceability'] = track_data['audio_features']['danceability']
                song_info[song_id]['tempo'] = track_data['audio_features']['tempo']
                song_info[song_id]['loudness'] = track_data['audio_features']['loudness']
                song_info[song_id]['acousticness'] = track_data['audio_features']['acousticness']
                song_info[song_id]['instrumentalness'] = track_data['audio_features']['instrumentalness']
                song_info[song_id]['liveness'] = track_data['audio_features']['liveness']
                song_info[song_id]['speechiness'] = track_data['audio_features']['speechiness']
                song_info[song_id]['duration_ms'] = track_data['audio_features']['duration_ms']  
                song_info[song_id]['popularity'] = track_data['popularity']
                song_info[song_id]['time_signature'] = track_data['audio_features']['time_signature']
                song_info[song_id]['artist_popularity'] = track_data['artist_popularity']
            else:
                # Set attributes to None if there's no data
                song_info[song_id]['energy'] = None
                song_info[song_id]['valence'] = None
                song_info[song_id]['danceability'] = None
                song_info[song_id]['tempo'] = None
                song_info[song_id]['loudness'] = None
                song_info[song_id]['acousticness'] = None
                song_info[song_id]['instrumentalness'] = None
                song_info[song_id]['liveness'] = None
                song_info[song_id]['speechiness'] = None
                song_info[song_id]['popularity'] = None
                song_info[song_id]['duration_ms'] = None
                song_info[song_id]['time_signature'] = None
                song_info[song_id]['artist_popularity'] = None

# Compute metrics
def max_metric(metric):
    return max(metric.values())

# Function to calculate the minimum of a dictionary
def min_metric(metric):
    return min(metric.values())

# Function to calculate the average of a dictionary
def avg_metric(metric):
    return sum(metric.values()) / len(metric)

# Function to calculate the median of a dictionary
def median_metric(metric):
    sorted_metric = sorted(metric.values())
    n = len(sorted_metric)
    if n % 2 == 0:
        return (sorted_metric[n//2] + sorted_metric[n//2 - 1]) / 2
    return sorted_metric[n//2]

degree = dict(G.degree())
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G)
clustering_coefficient = nx.clustering(G)

# Calculate average metrics
avg_degree = avg_metric(degree)
avg_degree_centrality = avg_metric(degree_centrality)
avg_betweenness_centrality = avg_metric(betweenness_centrality)
avg_closeness_centrality = avg_metric(closeness_centrality)
avg_eigenvector_centrality = avg_metric(eigenvector_centrality)
avg_path_length = nx.average_shortest_path_length(G)
avg_clustering_coefficient = nx.average_clustering(G)

# Calculate standard deviation metrics
std_degree = np.std(list(degree.values()))
std_degree_centrality = np.std(list(degree_centrality.values()))
std_betweenness_centrality = np.std(list(betweenness_centrality.values()))
std_closeness_centrality = np.std(list(closeness_centrality.values()))
std_eigenvector_centrality = np.std(list(eigenvector_centrality.values()))
std_clustering_coefficient = np.std(list(clustering_coefficient.values()))

# Calculate median metrics
median_degree = median_metric(degree)
median_degree_centrality = median_metric(degree_centrality)
median_betweenness_centrality = median_metric(betweenness_centrality)
median_closeness_centrality = median_metric(closeness_centrality)
median_eigenvector_centrality = median_metric(eigenvector_centrality)
median_clustering_coefficient = median_metric(clustering_coefficient)

# Calculate max metrics
max_degree = max_metric(degree)
max_degree_centrality = max_metric(degree_centrality)
max_betweenness_centrality = max_metric(betweenness_centrality)
max_closeness_centrality = max_metric(closeness_centrality)
max_eigenvector_centrality = max_metric(eigenvector_centrality)
max_clustering_coefficient = max_metric(clustering_coefficient)

# Calculate min metrics
min_degree = min_metric(degree)
min_degree_centrality = min_metric(degree_centrality)
min_betweenness_centrality = min_metric(betweenness_centrality)
min_closeness_centrality = min_metric(closeness_centrality)
min_eigenvector_centrality = min_metric(eigenvector_centrality)
min_clustering_coefficient = min_metric(clustering_coefficient)

# Danceability
average_danceability = avg_metric({k: float(v['danceability']) for k, v in song_info.items()})

average_danceability = avg_metric({k: float(v['danceability']) for k, v in song_info.items()})
average_energy = avg_metric({k: float(v['energy']) for k, v in song_info.items()})
average_valence = avg_metric({k: float(v['valence']) for k, v in song_info.items()})
average_tempo = avg_metric({k: float(v['tempo']) for k, v in song_info.items()})
average_loudness = avg_metric({k: float(v['loudness']) for k, v in song_info.items()})
average_acousticness = avg_metric({k: float(v['acousticness']) for k, v in song_info.items()})
average_instrumentalness = avg_metric({k: float(v['instrumentalness']) for k, v in song_info.items()})
average_liveness = avg_metric({k: float(v['liveness']) for k, v in song_info.items()})
average_speechiness = avg_metric({k: float(v['speechiness']) for k, v in song_info.items()})
average_duration_ms = avg_metric({k: float(v['duration_ms']) for k, v in song_info.items()})
average_popularity = avg_metric({k: float(v['popularity']) for k, v in song_info.items()})
average_time_signature = avg_metric({k: float(v['time_signature']) for k, v in song_info.items()})

# Draw unweighted graph
def draw_unweighted(G, pos, node_color, edge_color, linewidths, border_color):
    node_size = 200
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_size, edgecolors=border_color , linewidths=linewidths)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, alpha=0.5)
    plt.title("Unweighted") 
    plt.show()

# Draw weighted graph by degree
def draw_degree(G, pos, node_color, edge_color, linewidths, border_color, degree, max_degree, song_info):
    node_size = [(degree[node] * 4 + 20) for node in G.nodes()]
    labels = {}
    labels[max_degree] = song_info[max_degree]['song_name']

    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_size, edgecolors=border_color, linewidths=linewidths)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, alpha=0.5)
    nx.draw_networkx_labels(G, pos, labels, font_size=3, font_weight='bold')
    plt.title("Degree")
    plt.show()

# Draw weighted graph by degree centrality
def draw_degree_centrality(G, pos, node_color, edge_color, linewidths, border_color, degree_centrality, song_info):
    node_size = [(((degree_centrality[node])* 60) ** 3 + 20) for node in G.nodes()]
    labels = {}
    max_degree_centrality = max(degree_centrality, key=degree_centrality.get)
    labels[max_degree_centrality] = song_info[max_degree_centrality]['song_name']

    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_size, edgecolors=border_color, linewidths=linewidths)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, alpha=0.5)
    nx.draw_networkx_labels(G, pos, labels, font_size=3, font_weight='bold')
    plt.title("Degree Centrality")
    plt.show()

# Draw weighted graph by betweenness centrality
def draw_betweenness_centrality(G, pos, node_color, edge_color, linewidths, border_color, betweenness_centrality, song_info):
    node_size = [(((betweenness_centrality[node] + 0.0001)* 200) ** 2 + 20) for node in G.nodes()]
    labels = {}
    max_betweenness_centrality = max(betweenness_centrality, key=betweenness_centrality.get)
    labels[max_betweenness_centrality] = song_info[max_betweenness_centrality]['song_name']

    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_size, edgecolors=border_color, linewidths=linewidths)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, alpha=0.5)
    nx.draw_networkx_labels(G, pos, labels, font_size=3, font_weight='bold')
    plt.title("Betweenness Centrality")
    plt.show()

# Draw weighted graph by closeness centrality
def draw_closeness_centrality(G, pos, node_color, edge_color, linewidths, border_color, closeness_centrality, song_info):
    node_size = [((closeness_centrality[node] * 12) ** 4) for node in G.nodes()]
    labels = {}
    max_closeness_centrality = max(closeness_centrality, key=closeness_centrality.get)
    labels[max_closeness_centrality] = song_info[max_closeness_centrality]['song_name']

    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_size, edgecolors=border_color, linewidths=linewidths)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, alpha=0.5)
    nx.draw_networkx_labels(G, pos, labels, font_size=3, font_weight='bold')
    plt.title("Closeness Centrality")
    plt.show()

# Draw weighted graph by eigenvector centrality
def draw_eigenvector_centrality(G, pos, node_color, edge_color, linewidths, border_color, eigenvector_centrality, song_info):
    node_size = [((eigenvector_centrality[node] * 1200) ** 1.2 + 60) for node in G.nodes()]
    labels = {}
    max_eigenvector_centrality = max(eigenvector_centrality, key=eigenvector_centrality.get)
    labels[max_eigenvector_centrality] = song_info[max_eigenvector_centrality]['song_name']

    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_size, edgecolors=border_color, linewidths=linewidths)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, alpha=0.5)
    nx.draw_networkx_labels(G, pos, labels, font_size=3, font_weight='bold')
    plt.title("Eigenvector Centrality")
    plt.show()

def draw_artist_popualrity(G, pos, node_color, edge_color, linewidths, border_color, song_info):
    artist_popularity = {node: song_info[node]['artist_popularity'] for node in G.nodes()}
    node_size = [(artist_popularity[node] * 2 + 20) for node in G.nodes()]
    labels = {}
    max_artist_popularity = max(artist_popularity, key=artist_popularity.get)
    labels[max_artist_popularity] = song_info[max_artist_popularity]['song_name']

    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_size, edgecolors=border_color, linewidths=linewidths)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, alpha=0.5)
    nx.draw_networkx_labels(G, pos, labels, font_size=3, font_weight='bold')
    plt.title("Artist Popularity")
    plt.show()

def draw_danceability(G, pos, node_color, edge_color, linewidths, border_color, song_info):
    danceability = {node: song_info[node]['danceability'] for node in G.nodes()}
    # normalize the danceability values
    #danceability = {k: (v - min(danceability.values())) / (max(danceability.values()) - min(danceability.values())) for k, v in danceability.items()}
    node_size = [((danceability[node] * 200) **2 ) for node in G.nodes()]
    labels = {}
    max_danceability = max(danceability, key=danceability.get)
    labels[max_danceability] = song_info[max_danceability]['song_name']

    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_size, edgecolors=border_color, linewidths=linewidths)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, alpha=0.5)
    nx.draw_networkx_labels(G, pos, labels, font_size=3, font_weight='bold')
    plt.title("Danceability")
    plt.show()

    # Draw the network    
node_colors = 'darkred'
border_color = 'black'
edge_color = 'gray'
linewidths = 0.5
labels = {}
largest_node = max(degree, key=degree.get)
labels[largest_node] = song_info[largest_node]['song_name']

pos = nx.spring_layout(G, k=3, iterations=300)
draw_danceability(G, pos, node_colors, edge_color, linewidths, border_color, song_info)
print("Average Danceability: ", average_danceability)