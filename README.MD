# Setup
Create a python virtual environment and install all dependencies in the requirements.txt file.
Use python3 (recommended to use python3.8)

With pip you can install using `pip install -r requirements.txt`

# Running
A list of possible configurations and experiments

## Running PPR on an example graph
Command to run PPR on the movies dataset
```
python main.py \
--file_path Data/Movies/triples.csv \
--source head_uri --target tail_uri \
--edge_attr relation \
--graph_output_dir graphs/Movies/ \
--output_dir output/Movies/ \
--num_q_nodes 5 \
--print_node_names_in_top_k \
--seed 1
```

## Testing PPR with particle filtering from a single source node
Here we run PPR from a single source node. If we consider multiple query nodes then get a combined ppr vector by running ppr once from each of the query nodes.

```
python main.py \
--graph_path graphs/Movies/graph.gpickle \
--output_dir output/Movies/ \
--num_q_nodes 5 \
--seed 1 \
--run_ppr_from_each_query_node
```

## Running on the larger dbpedia dataset

```
python main.py \
--graph_path graphs/dbpedia/graph.gpickle \
--output_dir output/dbpedia/ \
--num_q_nodes 5 \
--seed 1 \
```

## Running PPR with specific query nodes

### Example 1
The following command will find nodes most proximal to the following node:
* http://www.wikidata.org/entity/Q8877 -> Steven Spielberg
```
python main.py \
--graph_path graphs/Movies/graph.gpickle \
--output_dir output/Movies/ \
--user_specified_query_nodes http://www.wikidata.org/entity/Q8877 http://www.wikidata.org/entity/Q188473 \
--print_node_names_in_top_k --run_networkx_ppr
```

### Example 2
The following command will find nodes most proximal to the following nodes:
* http://www.wikidata.org/entity/Q8877 -> Steven Spielberg
* http://www.wikidata.org/entity/Q188473 -> Action Film Genre
```
python main.py \
--graph_path graphs/Movies/graph.gpickle \
--output_dir output/Movies/ \
--user_specified_query_nodes http://www.wikidata.org/entity/Q8877 http://www.wikidata.org/entity/Q188473 \
--print_node_names_in_top_k --run_networkx_ppr
```

### Example 3:
The following command will find nodes most proximal to the following nodes:
* http://www.wikidata.org/entity/Q8877 -> Steven Spielberg
* http://www.wikidata.org/entity/Q4465 -> Peter Jackson
* http://www.wikidata.org/entity/Q167726 -> Jurassic Park
* http://www.wikidata.org/entity/Q127367 -> Lord of the Rings: The Fellowship of the Ring
```
python main.py \
--graph_path graphs/Movies/graph.gpickle \
--output_dir output/Movies/ \
--user_specified_query_nodes http://www.wikidata.org/entity/Q8877 \
http://www.wikidata.org/entity/Q4465 http://www.wikidata.org/entity/Q167726 \
http://www.wikidata.org/entity/Q127367 \
--print_node_names_in_top_k --run_networkx_ppr
```

### Example 4:
The following command will find nodes most proximal to the following nodes:
* http://www.wikidata.org/entity/Q8877 -> Steven Spielberg
* http://www.wikidata.org/entity/Q4465 -> Peter Jackson
```
python main.py \
--graph_path graphs/Movies/graph.gpickle \
--output_dir output/Movies/ \
--user_specified_query_nodes http://www.wikidata.org/entity/Q8877 \
http://www.wikidata.org/entity/Q4465 \
--print_node_names_in_top_k --run_networkx_ppr
```

Also can execute `./ppr_from_each_query_node.sh` shell script to run with 10 different seeds and observe trend

# Datasets

## Movies
Knowledge graph taken from https://mindreader.tech/dataset/. File is `triples.csv`