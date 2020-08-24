#plots for rq1
python gen_plot.py -t 1 -o ./rq1-adult-tree.png ./results/at1/*.pkl
python gen_plot.py -t 1 -o ./rq1-german-tree.png ./results/gt/*.pkl
python gen_plot.py -t 1 -o ./rq1-adult-forest.png ./results/af1/*.pkl
python gen_plot.py -t 1 -o ./rq1-german-forest.png ./results/gf1/*.pkl

# plots for rq2
python gen_plot.py -t 2 -o ./rq2-adult-tree.png ./results/at1/*.f.0.8.s.1.*.pkl
python gen_plot.py -t 2 -o ./rq2-adult-forest.png ./results/af1/*.f.0.8.s.1.*.pkl

# plots for rq3
python gen_plot.py -t 5 -o ./rq3-adult-tree.png ./results/at2/*.pkl
python gen_plot.py -t 5 -o ./rq3-adult-forest.png ./results/af3/*.pkl

# plots for rq4
python gen_plot.py -t 13 -o ./rq4-adult-forest.png ./results/af3/*.s.3.*.pkl

# plots for rq5
python gen_plot.py -t 14 -o ./rq5-adult-tree.png ./results/at1/*.pkl
python gen_plot.py -t 14 -o ./rq5-adult-forest.png ./results/af1/*.pkl
