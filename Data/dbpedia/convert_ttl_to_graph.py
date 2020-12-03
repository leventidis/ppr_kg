import networkx as nx
import pandas as pd
import pickle

from tqdm import tqdm

def save_data(data, output_dir, file_name):
    with open(output_dir + file_name, 'wb') as handle:
        pickle.dump(data, handle)

def create_id_to_node_dict(G):
    print('Creating id_to_node_dict...')
    # Convert all node names to integer IDs (starting with ID=0)
    ids = range(G.number_of_nodes())
    nodes = list(G.nodes())
    id_to_node_dict = {ids[i]: nodes[i] for i in range(len(ids))}

    save_data(data=id_to_node_dict, output_dir=output_dir, file_name='id_to_node_dict.pickle')

def create_node_to_id_dict(G):
    print('Creating node_to_id_dict...')
    # Convert all node names to integer IDs (starting with ID=0)
    ids = range(G.number_of_nodes())
    nodes = list(G.nodes())
    node_to_id_dict = {nodes[i]: ids[i] for i in range(G.number_of_nodes())}

    save_data(data=node_to_id_dict, output_dir=output_dir, file_name='node_to_id_dict.pickle')

    print('Re-labelling graph')
    G = nx.relabel_nodes(G, node_to_id_dict)
    print('Created graph has:', G.number_of_nodes(), 'nodes and', G.number_of_edges(), 'edges.')

    # Save the graph
    print('Saving graph with integer ids and dictionaries...')
    nx.write_gpickle(G, path=output_dir + 'graph.gpickle')

ttl_file = 'mappingbased-objects_lang=en.ttl'

df = pd.read_csv(ttl_file, sep=' ', index_col=False, usecols=['head_uri', 'relation', 'tail_uri'], nrows=10000000)
print('Finished reading dataframe')

print(df.shape)

source = 'head_uri'
target = 'tail_uri'
edge_attr = 'relation'
output_dir = '../../graphs/dbpedia/'

G = nx.from_pandas_edgelist(df, source=source, target=target, edge_attr=edge_attr, create_using=nx.DiGraph())
print('Finished creating networkx graph')

print('Adding weights to the edges...')
# Add weights to the edges
for node in tqdm(G):
    edges = list(G.out_edges(node, data=True))
    if len(edges) >= 1:
        # Assign a uniform weight to each outgoing edge such that all outgoing edges have weights that sum to 1
        weight = 1 / len(edges)
        for edge in edges:
            G[edge[0]][edge[1]]['weight'] = weight

print('Saving graph')
nx.write_gpickle(G, path=output_dir + 'graph.gpickle')

# Create dictionaries and re-label the graph
create_id_to_node_dict(G)
create_node_to_id_dict(G)