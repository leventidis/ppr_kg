import networkx as nx
import pandas as pd
import pickle

ttl_file = 'mappingbased-objects_lang=en.ttl'

df = pd.read_csv(ttl_file, sep=' ', index_col=False, usecols=['head_uri', 'relation', 'tail_uri'], nrows=5000000)
print('Finished reading dataframe')

print(df.shape)

source = 'head_uri'
target = 'tail_uri'
edge_attr = 'relation'
output_dir = '../../graphs/dbpedia/'

G = nx.from_pandas_edgelist(df, source=source, target=target, edge_attr=edge_attr, create_using=nx.DiGraph())
print('Finished creating networkx graph')

# Add a weight of 1 to every edge
nx.set_edge_attributes(G, 1, 'weight')

print('Saving graph')
nx.write_gpickle(G, path=output_dir + 'graph.gpickle')

print('Creating integer ids...')
# Convert all node names to integer IDs (starting with ID=0)
ids = range(G.number_of_nodes())
nodes = list(G.nodes())
id_to_node_dict = {ids[i]: nodes[i] for i in range(len(ids))}
node_to_id_dict = {nodes[i]: ids[i] for i in range(G.number_of_nodes())}
G = nx.relabel_nodes(G, node_to_id_dict)

# Save the graph and the dictionaries
print('Saving graph with integer ids and dictionaries...')
nx.write_gpickle(G, path=output_dir + 'graph.gpickle')
with open(output_dir + 'id_to_node_dict.pickle', 'wb') as handle:
    pickle.dump(id_to_node_dict, handle)
with open(output_dir + 'node_to_id_dict.pickle', 'wb') as handle:
    pickle.dump(node_to_id_dict, handle)
