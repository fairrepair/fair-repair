# Fairness threshold list
declare -a FairnessThreshList=(0.5 0.6 0.7 0.8 0.9 0.95)

# Alpha values list
declare -a AlphaList=(2.0 1.8 1.6 1.4 1.2 1.05)

# Random seeds list
declare -a RandSeedList=(1 2 3 4 5)

# Dataset list
declare -a DataList=("german.data")

# The sensitive attribute inputs list
# (Be careful with the formatting and copy each line exactly)
declare -a SensList=(
"['Sex','ForeignWorker']" \
"['Sex']" \
"['ForeignWorker']" \
)

echo -n "Starting: " && date

# Nested loops for getting all combinations of parameters
for fair in ${FairnessThreshList[@]}; do
    for alpha in ${AlphaList[@]}; do
        for seed in ${RandSeedList[@]}; do

            # Do not reorder the inner sensitive loop
            dataID=0
            for data in ${DataList[@]}; do
                dataID=$((dataID+1))

                sensID=0
                for sens in ${SensList[@]}; do
                    sensID=$((sensID+1))
                
                    # define the base output filename in terms of all the parameters
                    outf="eval.german.r.$seed.a.$alpha.f.$fair.s.$sensID.i.$dataID"
                    echo "START ------------------------------------------------------"
                    
                    # define the command with all the parameters as a string variable
                    CMD="-r $seed -a $alpha -f $fair -s $sens -i $data --eval-file-output ./gt/${outf}.pkl"
                    
                    # echo the command we will run: very useful for debugging
                    echo "Running python german_patch.py with args: $CMD" 

                    # RUN the command
                    python3 ./german_patch.py $CMD &> ./gt/${outf}.eval
                    
                    echo "DONE ------------------------------------------------------"
                done
            done
        done
    done
done

echo -n "Finished: " && date

