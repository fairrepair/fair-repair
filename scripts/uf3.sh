# Fairness threshold list
declare -a FairnessThreshList=(0.8)

# Alpha values list
declare -a AlphaList=(1.2)

# Forest size list
declare -a ForestSize=(10 20 30 40 50 60 70 80 90 100)

# Random seeds list
declare -a RandSeedList=(1 2 3 4 5)

# Dataset list
declare -a DataList=(
# "ufrgs.data.5000" \
# "ufrgs.data.10000" \
# "ufrgs.data.15000" \
# "ufrgs.data.20000" \
# "ufrgs.data.25000" \
# "ufrgs.data.30000" \
# "ufrgs.data.35000" \
# "ufrgs.data.40000" \
"ufrgs.data" \
)

echo -n "Starting: " && date

# Nested loops for getting all combinations of parameters
for fair in ${FairnessThreshList[@]}; do
    for alpha in ${AlphaList[@]}; do
        for seed in ${RandSeedList[@]}; do
            for size in ${ForestSize[@]}; do

                # Do not reorder the inner sensitive loop
                dataID=0
                for data in ${DataList[@]}; do
                    dataID=$((dataID+1))

                    # define the base output filename in terms of all the parameters
                    outf="eval.ufrgs.r.$seed.a.$alpha.f.$fair.i.$dataID"
                    echo "START ------------------------------------------------------"
                    
                    # define the command with all the parameters as a string variable
                    CMD="-r $seed -a $alpha -f $fair -i $data -t $size --eval-file-output ./results/uf3/${outf}.pkl"
                    
                    # echo the command we will run: very useful for debugging
                    echo "Running python ufrgs_patch.py with args: $CMD" 

                    # RUN the command
                    python ./ufrgs_patch.py $CMD &> ./results/uf3/${outf}.eval
                    
                    echo "DONE ------------------------------------------------------"
                done
            done
        done
    done
done

echo -n "Finished: " && date

