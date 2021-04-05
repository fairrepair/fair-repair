#!/bin/bash

#####################
# Generates randomly (uniform) sampled partitions of the file
# adult.data.noheader.
#####################

declare -a FilePartitions=(5000 10000 15000 20000 25000 30000 35000 40000 45000) 

echo -n "Starting: " && date

for part in ${FilePartitions[@]}; do
    echo "Generating $part-line file from adult dataset..."
    fname="adult.data.$part"
    cat adult.header > $fname
    shuf -n $part ./adult.data.noheader >> $fname
done

echo -n "Finished: " && date
