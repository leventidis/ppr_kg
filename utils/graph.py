import networkx as nx
import pandas as pd

def get_graph_from_csv(file, source, target, edge_attr):
    '''
    Reads the specified csv file into a pandas dataframe and constructs a directed graph
    based on the specified source, target arguments and attributes.

    The edges of the graph also have a weight. For now the weight is 1 for all edges
    TODO: Allow for non-uniform edge weights.

    Arguments
    -------
        file (str): Path to the input csv file. The CSV file will be converted into a pandas dataframe

        source (str or int): A valid column name (string or integer) for the source nodes

        target (str or int): A valid column name (string or integer) for the target nodes

        edge_attr (str or int, iterable): A valid column name (str or int) or iterable of 
        column names that are used to retrieve items and add them to the graph as edge attributes.

    Returns
    -------
    A directed networkx graph
    '''

    df = pd.read_csv(file)
    G = nx.from_pandas_edgelist(df, source=source, target=target, edge_attr=edge_attr, create_using=nx.DiGraph())

    # Add weights to the edges
    for node in G:
        edges = list(G.out_edges(node, data=True))
        # Assign a uniform weight to each outgoing edge such that all outgoing edges have weights that sum to 1
        weight = 1 / len(edges)
        for edge in edges:
            G[edge[0]][edge[1]]['weight'] = weight

    return G