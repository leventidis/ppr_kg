#!/bin/bash

# Shell script to run multiple times with different seeds the ppr from a single node

graph_path=graphs/random/nodes500000/
output_dir=output/random/nodes500000/

# Sizes of query sets to test
declare -a num_query_nodes_arr=("5" "10" "20" "100")

# Number of runs (i.e. distict seeds) at each query set size
num_runs=5

for i in "${num_query_nodes_arr[@]}"
do
    num_q_nodes=$i
    for cur_run in $(seq 1 $num_runs);
    do
        seed=$cur_run

        python main.py \
            --graph_path "$graph_path"graph.gpickle \
            --output_dir "${output_dir}query_nodes_${num_q_nodes}/seed$seed/" \
            --num_q_nodes $num_q_nodes \
            --seed $seed \
            --no_dictionary \
            --run_ppr_from_each_query_node
            # --distributed_single_source_ppr
    done
done