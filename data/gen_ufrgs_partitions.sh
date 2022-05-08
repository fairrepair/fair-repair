#!/bin/bash

#########################################################
# Generates randomly (uniform) sampled partitions of the 
# file ufrgs.data.noheader.
#########################################################

declare -a FilePartitions=(5000 10000 15000 20000 25000 30000 35000 40000) 

echo -n "Starting: " && date

for part in ${FilePartitions[@]}; do
    echo "Generating $part-line file from ufrgs dataset..."
    fname="ufrgs.data.$part"
    cat ufrgs.header > $fname
    shuf -n $part ./ufrgs.data.noheader >> $fname
done

echo -n "Finished: " && date
