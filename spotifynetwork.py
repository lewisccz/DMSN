import pickle
import networkx as nx 
import matplotlib.pyplot as plt
import csv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

def main():
    client_id = '736dd37c127e4fe698338b94359c87b1'
    client_secret = 'e2f427be0bae4ae6aef0e5945d871dfc'

    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

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

    # Add the 'energy' audio feature to each song
    for song_id in song_info:
        audio_features = sp.audio_features(song_id)[0]
        if audio_features:
            song_info[song_id]['energy'] = audio_features['energy']
        else:
            song_info[song_id]['energy'] = None

    # Compute metrics
    degree = dict(G.degree())
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)
    clustering_coefficient = nx.clustering(G)

    # Calculate average metrics
    avg_degree = avg_metric(degree)
    avg_degree_centrality = avg_metric(degree_centrality)
    avg_betweenness_centrality = avg_metric(betweenness_centrality)
    avg_closeness_centrality = avg_metric(closeness_centrality)
    avg_eigenvector_centrality = avg_metric(eigenvector_centrality)
    avg_path_length = nx.average_shortest_path_length(G)
    avg_clustering_coefficient = nx.average_clustering(G)

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

    # Iterate over the nodes in the network
    for node in G.nodes():
        print(f"Node: {node}")
        
        # Access the attributes of the node
        attributes = G.nodes[node]
        
        # Print the attributes
        for attr_name, attr_value in attributes.items():
            print(f"{attr_name}: {attr_value}")
        
        # Access additional song information from the CSV file
        song_data = song_info[node]
        print(f"Song Name: {song_data['song_name']}")
        print(f"Artist Name: {song_data['artist_name']}")
        print(f"Artist URI: {song_data['artist_uri']}")
        print(f"Album Name: {song_data['album_name']}")
        print(f"Album URI: {song_data['album_uri']}")
        print("Energy:", song_data['energy'])
        print(f"Degree: {degree[node]}")
        print(f"Degree Centrality: {degree_centrality[node]}")
        print(f"Betweenness Centrality: {betweenness_centrality[node]}")
        print(f"Closeness Centrality: {closeness_centrality[node]}")
        print(f"Eigenvector Centrality: {eigenvector_centrality[node]}")
        print(f"Clustering Coefficient: {clustering_coefficient[node]}")
        print("---")

    # Print the metrics
    print("Max degree:", max_degree, "Node:", max(degree, key=degree.get), "Song Name:", song_info[max(degree, key=degree.get)]['song_name']) 
    print("Min degree:", min_degree, "Node:", min(degree, key=degree.get), "Song Name:", song_info[min(degree, key=degree.get)]['song_name']) 
    print("Average degree:", avg_degree)
    print("Median degree:", median_degree)
    print("---")

    print("Max degree centrality:", max_degree_centrality, "Node:", max(degree_centrality, key=degree_centrality.get), "Song Name:", song_info[max(degree_centrality, key=degree_centrality.get)]['song_name'])
    print("Min degree centrality:", min_degree_centrality, "Node:", min(degree_centrality, key=degree_centrality.get), "Song Name:", song_info[min(degree_centrality, key=degree_centrality.get)]['song_name'])
    print("Average degree centrality:", avg_degree_centrality)
    print("Median degree centrality:", median_degree_centrality)
    print("---")

    print("Max betweenness centrality:", max_betweenness_centrality, "Node:", max(betweenness_centrality, key=betweenness_centrality.get), "Song Name:", song_info[max(betweenness_centrality, key=betweenness_centrality.get)]['song_name'])
    print("Min betweenness centrality:", min_betweenness_centrality, "Node:", min(betweenness_centrality, key=betweenness_centrality.get), "Song Name:", song_info[min(betweenness_centrality, key=betweenness_centrality.get)]['song_name'])
    print("Average betweenness centrality:", avg_betweenness_centrality)
    print("Median betweenness centrality:", median_betweenness_centrality)
    print("---")

    print("Max closeness centrality:", max_closeness_centrality, "Node:", max(closeness_centrality, key=closeness_centrality.get), "Song Name:", song_info[max(closeness_centrality, key=closeness_centrality.get)]['song_name'])
    print("Min closeness centrality:", min_closeness_centrality, "Node:", min(closeness_centrality, key=closeness_centrality.get), "Song Name:", song_info[min(closeness_centrality, key=closeness_centrality.get)]['song_name'])
    print("Average closeness centrality:", avg_closeness_centrality)
    print("Median closeness centrality:", median_closeness_centrality)
    print("---")

    print("Max eigenvector centrality:", max_eigenvector_centrality, "Node:", max(eigenvector_centrality, key=eigenvector_centrality.get), "Song Name:", song_info[max(eigenvector_centrality, key=eigenvector_centrality.get)]['song_name'])
    print("Min eigenvector centrality:", min_eigenvector_centrality, "Node:", min(eigenvector_centrality, key=eigenvector_centrality.get), "Song Name:", song_info[min(eigenvector_centrality, key=eigenvector_centrality.get)]['song_name'])
    print("Average eigenvector centrality:", avg_eigenvector_centrality)
    print("Median eigenvector centrality:", median_eigenvector_centrality)
    print("---")

    print("Max clustering coefficient:", max_clustering_coefficient, "Node:", max(clustering_coefficient, key=clustering_coefficient.get), "Song Name:", song_info[max(clustering_coefficient, key=clustering_coefficient.get)]['song_name'])
    print("Min clustering coefficient:", min_clustering_coefficient, "Node:", min(clustering_coefficient, key=clustering_coefficient.get), "Song Name:", song_info[min(clustering_coefficient, key=clustering_coefficient.get)]['song_name'])
    print("Average clustering coefficient:", avg_clustering_coefficient)
    print("Median clustering coefficient:", median_clustering_coefficient)
    print("---")

    print("Average path length:", avg_path_length) 
    print("---")

    # Draw the network    
    node_colors = 'darkred'
    border_color = 'black'
    edge_color = 'gray'
    linewidths = 0.5
    labels = {}
    largest_node = max(degree, key=degree.get)
    labels[largest_node] = song_info[largest_node]['song_name']

    pos = nx.spring_layout(G, k=3, iterations=300)
    draw_unweighted(G, pos, node_colors, edge_color, linewidths, border_color)
    #draw_degree(G, pos, node_colors, edge_color, linewidths, border_color, degree, largest_node, song_info)
    #draw_degree_centrality(G, pos, node_colors, edge_color, linewidths, border_color, degree_centrality, song_info)
    #draw_betweenness_centrality(G, pos, node_colors, edge_color, linewidths, border_color, betweenness_centrality, song_info)
    #draw_closeness_centrality(G, pos, node_colors, edge_color, linewidths, border_color, closeness_centrality, song_info)
    #draw_eigenvector_centrality(G, pos, node_colors, edge_color, linewidths, border_color, eigenvector_centrality, song_info)

# Function to calculate the maximum of a dictionary
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

main()





