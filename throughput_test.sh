#!/bin/bash

# Shell script to check throuput of re-using query nodes with single source vs.
# traditional particle filtering with multiple sources

graph_path=graphs/Movies/graph.gpickle
output_dir=output/throughput_test/

# Number of runs (i.e. distict seeds) at each query set size
num_runs=50
num_q_nodes=5

for cur_run in $(seq 1 $num_runs);
do
    seed=$cur_run

    python main.py \
        --graph_path $graph_path \
        --output_dir "${output_dir}"run_${cur_run}/ \
        --num_q_nodes $num_q_nodes \
        --seed $seed \
        --no_dictionary
done

echo "Running distributed single source..."
num_q_nodes=200

python main.py \
    --graph_path $graph_path \
    --output_dir "${output_dir}"single_source/ \
    --num_q_nodes $num_q_nodes \
    --seed 1 \
    --no_dictionary \
    --run_ppr_from_each_query_node \
    --distributed_single_source_ppr