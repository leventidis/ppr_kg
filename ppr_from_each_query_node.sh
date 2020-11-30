#!/bin/bash

# Shell script to run multiple times with different seeds the ppr from a single node

file_path=Data/Movies/triples.csv
graph_output_dir=graphs/single_source_ppr_movies/
output_dir=output/single_source_ppr_movies/
num_q_nodes=10

num_runs=10

for cur_run in $(seq 1 $num_runs);
do
    seed=$cur_run

    python main.py --file_path $file_path --source head_uri --target tail_uri \
        --edge_attr relation \
        --graph_output_dir "${graph_output_dir}seed$seed/" \
        --output_dir "${output_dir}seed$seed/" \
        --num_q_nodes $num_q_nodes \
        --seed $seed \
        --run_ppr_from_each_query_node
done