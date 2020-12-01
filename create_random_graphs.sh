#!/bin/bash

# Shell to create multiple random graphs 

output_dir=graphs/random/
graph_generator=erdos-renyi

declare -a num_nodes_arr=("1000" "5000" "10000" "25000" "100000" "200000" "500000" "1000000")

for i in "${num_nodes_arr[@]}"
do
    num_nodes=$i

    python random_graph_creator.py \
        --output_dir "${output_dir}nodes$i/" \
        --num_nodes $num_nodes \
        --graph_generator $graph_generator \
        --seed 1
done