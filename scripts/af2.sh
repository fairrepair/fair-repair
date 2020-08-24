# Fairness threshold list
declare -a FairnessThreshList=(0.8)

# Alpha values list
declare -a AlphaList=(1.2)

# Forest size list
declare -a ForestSize=(30)

# Random seeds list
declare -a RandSeedList=(1 2 3 4 5)

# Dataset list
declare -a DataList=(
"adult.data.5000" \
"adult.data.10000" \
"adult.data.15000" \
"adult.data.20000" \
"adult.data.25000" \
"adult.data.30000" \
"adult.data.35000" \
"adult.data.40000" \
"adult.data.45000" \
"adult.data" \
)

# The sensitive attribute inputs list
# (Be careful with the formatting and copy each line exactly)
declare -a SensList=(
"['sex']" \
"['race']" \
"['sex','race']" \
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

                    sensID=0
                    for sens in ${SensList[@]}; do
                        sensID=$((sensID+1))
                    
                        # define the base output filename in terms of all the parameters
                        outf="eval.adult.r.$seed.a.$alpha.f.$fair.s.$sensID.i.$dataID.t.$size"
                        echo "START ------------------------------------------------------"
                        
                        # define the command with all the parameters as a string variable
                        CMD="-r $seed -a $alpha -f $fair -s $sens -i $data -t $size --eval-file-output ./af2/${outf}.pkl"
                        
                        # echo the command we will run: very useful for debugging
                        echo "Running python adult_patch.py with args: $CMD" 

                        # RUN the command
                        python3 ./adult_patch.py $CMD &> ./af2/${outf}.eval
                        
                        echo "DONE ------------------------------------------------------"
                    done
                done
            done
        done
    done
done

echo -n "Finished: " && date

