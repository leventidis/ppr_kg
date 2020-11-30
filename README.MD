# Setup
Create a python virtual environment and install all dependencies in the requirements.txt file.
Use python3 (recommended to use python3.8)

With pip you can install using `pip install -r requirements.txt`

# Running PPR on an example graph
Command to run PPR on the movies dataset
```
python main.py \
--file_path Data/Movies/triples.csv \
--source head_uri --target tail_uri \
--edge_attr relation \
--graph_output_dir graphs/Movies/ \
--output_dir output/Movies/ \
--num_q_nodes 5 \
--seed 1
```

# Testing PPR with particle filtering from a single source node
Here we run PPR from a single source node. If we consider multiple query nodes then get a combined ppr vector by running ppr once from each of the query nodes.

```
python main.py \
--file_path Data/Movies/triples.csv \
--source head_uri --target tail_uri \
--edge_attr relation \
--graph_output_dir graphs/Movies/ \
--output_dir output/Movies/ \
--num_q_nodes 5 \
--seed 1 \
--run_ppr_from_each_query_node
```

Also can execute `./ppr_from_each_query_node.sh` shell script to run with 10 different seeds and observe trend

# Datasets

## Movies
Knowledge graph taken from https://mindreader.tech/dataset/. File is `triples.csv`