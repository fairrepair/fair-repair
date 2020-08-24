'''Script to generate plots, example commands:

Generate plot of type 1:
python3 gen_plot.py -t 1 -o out.pdf eval.german.r.1.a.2.0.f.0.5.pkl

Generate plot of type 2:
python3 gen_plot.py -t 2 -o out.pdf eval.german.r.1.a.2.0.f.0.5.pkl

Generate plot of type 3:
python3 gen_plot.py -t 3 -o out.pdf eval.german.r.1.a.2.0.f.0.5.pkl

All of the above output to out.pdf

All of the above read a single .pkl file: eval.german.r.1.a.2.0.f.0.5.pkl

But, you can specify multiple .pkl files and the plot commands will
read all of them and include all of their data. It's important to
specify those .pkl files that you want in the plot.
'''

import pprint
import argparse

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import statistics

import sys

# Adds higher directory to python modules path (to be able to import eutil)
sys.path.append("..") 
from eutil import *


def set_aspect_ratio(ax, ratio):
    # https://jdhao.github.io/2017/06/03/change-aspect-ratio-in-mpl/
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    # the abs method is used to make sure that all numbers are positive
    # because x and y axis of an axes maybe inversed.
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    return


def collectData(args,vals):
    w = []
    x = []
    y = []
    z = []
    a = []
    b = []
    c = []
    d = []
    e = []
    # f = {}
    # for i in [0,5,0.6,0.7,0.8,0.9,0.95]:
    #     f[i] = []

    for val in vals:
        # if val['alpha'] == 1.2:
        #     f['fairness-thresh'].append(1)
        w.append(val['accuracy-before'])
        a.append(val['precision-before'])
        b.append(val['recall-before'])
        x.append(val['accuracy-after'])
        c.append(val['precision-after'])
        d.append(val['recall-after'])
        y.append(val['fairness-thresh'])
        z.append(val['alpha'])
        # e.append(type(val['sensitive-attrs']))
        if val['sensitive-attrs'] == ['sex']:
            e.append(1)
        elif val['sensitive-attrs'] == ['race']:
            e.append(2)
        else:
            e.append(3)
    p = [y,z,e,w,a,b,x,c,d]
    for i in p:
        j = [str(element) for element in i]
        print(','.join(j))
    return


def plot1(args,vals):
    '''
    Plots a scatter plot of:
    x: target fairness 
    y: fairness achieved 
    '''
    fig, ax = plt.subplots()
    x = []
    y = []
    for val in vals:
        # if 'final-fairness' in val.keys():
        #     for foutput in val['final-fairness']:
        #         (f,g1,g2) = foutput
        #         x.append(val['fairness-thresh'])
        #         if f < val['fairness-thresh']:
        #             print(val['fairness-thresh'],val['alpha'],val['random-seed'],val['data-file'],val['sensitive-attrs'])
        #         y.append(f)
        # else:
        #     print(val['alpha'],val['random-seed'],val['data-file'],val['sensitive-attrs'])
        for foutput in val['final-fairness']:
            (f,g1,g2) = foutput
            x.append(val['fairness-thresh'])
            y.append(f)

    # print(x)
    # print(y)
    ax.scatter(x, y) #, c=close, s=volume, alpha=0.5)
    ax.set_xlabel(r'Target fairness', fontsize=15)
    ax.set_ylabel(r'Fairness achieved', fontsize=15)
    ax.set_title('')
    ax.grid(True)
    fig.tight_layout()


    # plot the Y=X line
    # https://stackoverflow.com/questions/25497402/adding-y-x-to-a-matplotlib-scatter-plot-if-i-havent-kept-track-of-all-the-data
    # lims = [
    #     np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    #     np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    # ]
    lims = [0.4,1.0]
    # now plot both limits against eachother
    bottombound, = ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0, label="y=x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    # fig.savefig('/Users/paul/Desktop/so.png', dpi=300)


    # plot the Y=1/X line
    # https://stackoverflow.com/questions/25497402/adding-y-x-to-a-matplotlib-scatter-plot-if-i-havent-kept-track-of-all-the-data
    def f(x):
        return 1/x
    fx_name = r'$f(x)=\frac{1}{x}$'

    x=np.setdiff1d(np.linspace(0.4,1.0,100),[0]) #to remove the zero
    y=f(x)
    topbound, = ax.plot(x, y, label="y=1/x",linestyle="--")

    # Set the x,y ranges
    axes = plt.gca()
    axes.set_xlim([0.4,1])
    axes.set_ylim([0.4, 2.25])

    # Set aspect ratio:
    # set_aspect_ratio(ax, 0.5)
    set_aspect_ratio(ax, 0.25)

    # axes.set_aspect('equal', adjustable='box')
    # axes.set_aspect(.25) #, adjustable='box')
    # ax.legend()
    plt.legend(handles=[topbound, bottombound])

    return fig


def plot2(args,vals):
    '''
    Plots scatter plot of:
    x: alpha
    y: semantic difference (number of points changed)
    '''
    fig, ax = plt.subplots()
    x = []
    y = []

    RAND_SEED = 1
    print("Only considering random-seed values of ", RAND_SEED)
        
    ############################
    # Plot the max sem difference points:
    x = []
    y = []
    for val in vals:
        if val['random-seed'] == RAND_SEED:
            hsets = val['init-num-hcubesets']
            alpha = val['alpha']
            print(alpha)
            x.append(alpha)
            
            # alpha*t.min is the theoretical min added up for all
            # hsets, and for each hsets, we need to make sure where is
            # no fraction, each hset must contain whole number of data
            # points, thus we add 1 for each hset
            y.append((val['theoretical-min-change'] * alpha) + hsets)
    # ax.scatter(x, y, label="Alpha bound", c="green", alpha=0.5)
    maxi, = ax.plot(x, y, label="Alpha bound", c="green", alpha=0.5, linestyle="--")

    ############################
    # Plot the dataset:
    x = []
    y = []
    for val in vals:
        if val['random-seed'] == RAND_SEED:
            x.append(val['alpha'])
            y.append(val['data-points-changed-total'])
    fair = ax.scatter(x, y, label="FairRepair", c="blue",alpha=0.5) #, c=close, s=volume, alpha=0.5)
    

    ############################
    # Plot the min sem difference points:
    x = []
    y = []
    for val in vals:
        if val['random-seed'] == RAND_SEED:
            x.append(val['alpha'])
            y.append(val['theoretical-min-change'])
    # ax.scatter(x, y, label="Min diff possible", c="red", alpha=0.5) #, c=close, s=volume, alpha=0.5)
    mini, = ax.plot(x, y, label="Min diff possible", c="red", alpha=0.5) #, c=close, s=volume, alpha=0.5)
    
    ax.set_xlabel(r'Alpha', fontsize=12)
    # ax.set_ylabel(r'Semantic difference (# points)', fontsize=15)
    ax.set_ylabel(r'Semantic diff.', fontsize=12)
    ax.set_title('')
    ax.grid(True)

    plt.legend(handles=[maxi, fair, mini])


    # Set aspect ratio:
    # set_aspect_ratio(ax, 0.5)
    set_aspect_ratio(ax, 0.25)

    fig.tight_layout()
    return fig    


def plot3(args,vals):
    '''
    x: number of data points in the dataset
    y: total runtime
    '''
    fig, ax = plt.subplots()
    

    fairness = {}
    for val in vals:
        f = val['fairness-thresh']
        x = val['forest-size']
        y = val['time-total']

        # if f < 0.7:
        #     continue

        # Group data by fairness f
        if f not in fairness:
            fairness[f] = {}
        if x not in fairness[f]:
            fairness[f][x] = []
        fairness[f][x].append(y)

    # pprint.pprint(fairness)
    # sys.exit(0)

    keys = list(fairness.keys())
    fairness_plt = {}
    for f in keys:
        fairness_plt[f] = {'x' : [], 'y' : []}
        for x in fairness[f]:
            median = statistics.median(fairness[f][x])
            fairness_plt[f]['x'].append(x)
            fairness_plt[f]['y'].append(median)

    # pprint.pprint(fairness_plt)
    # sys.exit(0)

    # plot
    keys = list(fairness_plt.keys())
    print(keys)
    keys.reverse()
    for f in keys:
        print(f)
        x = fairness_plt[f]['x']
        y = fairness_plt[f]['y']
        # ax.scatter(x, y, c=str(f), alpha = 0.5, label=f, cmap=cm.viridis)
        ax.scatter(x, y, c=str(f), label=f, cmap=cm.viridis)

    # x = []
    # y = []
    # for val in vals:
    #     if val['fairness-thresh'] == 0.8: #FAIRNESS_THRESH:
    #         x.append(val['num-input-points'])
    #         y.append(val['time-total'])
    # ax.scatter(x, y, c="red", label="Fairness=0.8") # close, s=volume, alpha=0.5)
    
    ax.set_xlabel(r'# of trees in forest', fontsize=15)
    ax.set_ylabel(r'Total time (s)', fontsize=15)
    ax.set_title('')
    
    # https://stackoverflow.com/questions/22239691/code-for-best-fit-straight-line-of-a-scatter-plot-in-python
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.polyfit.html
    # plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), linestyle="--", label="")
    # Returns a vector of coefficients p that minimises the squared error
    # plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), linestyle="--", label="Min square error 1D fit")
    # TODO this is better: http://seaborn.pydata.org/tutorial/regression.html#functions-to-draw-linear-regression-models

    # Set aspect ratio:
    # set_aspect_ratio(ax, 0.5)
    set_aspect_ratio(ax, 0.3)
    ax.grid(True)
    ax.legend(title="Fairness thresh")
    fig.tight_layout()
    return fig


def plot4(args,vals):
    '''
    x: number of paths in the input decisoin tree
    y: total runtime
    '''
    fig, ax = plt.subplots()
    fairness = {}
    for val in vals:
        f = val['fairness-thresh']
        x = val['num-hcubes-before']
        y = val['time-total']

        # if f < 0.7:
        #     continue

        # Group data by fairness f
        if f not in fairness:
            fairness[f] = {}
        if x not in fairness[f]:
            fairness[f][x] = []
        fairness[f][x].append(y)

    # pprint.pprint(fairness)
    # sys.exit(0)

    keys = list(fairness.keys())
    fairness_plt = {}
    for f in keys:
        fairness_plt[f] = {'x' : [], 'y' : []}
        for x in fairness[f]:
            median = statistics.median(fairness[f][x])
            fairness_plt[f]['x'].append(x)
            fairness_plt[f]['y'].append(median)

    #pprint.pprint(fairness_plt)
    # sys.exit(0)

    # plot
    keys = list(fairness_plt.keys())
    print(keys)
    keys.reverse()
    for f in keys:
        print(f)
        x = fairness_plt[f]['x']
        y = fairness_plt[f]['y']
        # ax.scatter(x, y, c=str(f), alpha = 0.5, label=f, cmap=cm.viridis)
        ax.scatter(x, y, c=str(f), label=f, cmap=cm.viridis)

    ax.set_xlabel(r'# of paths in the input tree', fontsize=15)
    ax.set_ylabel(r'Total time (s)', fontsize=15)
    ax.set_title('')

    # plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='City Area')
    ax.legend(title="Fairness thresh")

    # Set aspect ratio:
    set_aspect_ratio(ax, 0.5)

    ax.grid(True)
    fig.tight_layout()
    return fig


def plot5(args,vals):
    '''
    x: number of data points in the dataset
    y: total runtime
    '''
    fig, ax = plt.subplots(dpi=300)
    

    fairness = {}
    for val in vals:
        f = val['fairness-thresh']
        x = val['num-input-points']
        y = val['time-total']

        # if f < 0.7:
        #     continue

        # Group data by fairness f
        if f not in fairness:
            fairness[f] = {}
        if x not in fairness[f]:
            fairness[f][x] = []
        fairness[f][x].append(y)

    # pprint.pprint(fairness)
    # sys.exit(0)

    keys = list(fairness.keys())
    fairness_plt = {}
    for f in keys:
        fairness_plt[f] = {'x' : [], 'y' : []}
        for x in fairness[f]:
            median = statistics.median(fairness[f][x])
            fairness_plt[f]['x'].append(x)
            fairness_plt[f]['y'].append(median)

    # pprint.pprint(fairness_plt)
    # sys.exit(0)

    # plot
    keys = list(fairness_plt.keys())
    print(keys)
    keys.reverse()
    for f in keys:
        print(f)
        x = fairness_plt[f]['x']
        y = fairness_plt[f]['y']
        # ax.scatter(x, y, c=str(f), alpha = 0.5, label=f, cmap=cm.viridis)
        # ax.scatter(x, y,c='grey', label=f)
        ax.scatter(x, y,c=str(f), label=f, cmap=cm.viridis)
        

    # x = []
    # y = []
    # for val in vals:
    #     if val['fairness-thresh'] == 0.8: #FAIRNESS_THRESH:
    #         x.append(val['num-input-points'])
    #         y.append(val['time-total'])
    # ax.scatter(x, y, c="red", label="Fairness=0.8") # close, s=volume, alpha=0.5)
    
    ax.set_xlabel(r'# of points in dataset', fontsize=15)
    ax.set_ylabel(r'Total time (s)', fontsize=15)
    ax.set_title('')
    
    # https://stackoverflow.com/questions/22239691/code-for-best-fit-straight-line-of-a-scatter-plot-in-python
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.polyfit.html
    # plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), linestyle="--", label="")
    # Returns a vector of coefficients p that minimises the squared error
    # plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), linestyle="--", label="Min square error 1D fit")
    # TODO this is better: http://seaborn.pydata.org/tutorial/regression.html#functions-to-draw-linear-regression-models

    # Set aspect ratio:
    # set_aspect_ratio(ax, 0.5)
    set_aspect_ratio(ax, 0.25)
    ax.grid(True)
    ax.legend(title="Fairness thresh",fontsize=12,prop={'size': 8})
    fig.tight_layout()
    return fig


def plot6(args,vals):
    '''
    x: number of data points in the dataset
    y: total runtime
    '''
    fig, ax = plt.subplots()
    

    fairness = {}
    for val in vals:
        x = val['fairness-thresh']
        f = val['fairness-thresh']
        y = val['time-total']

        # if f < 0.7:
        #     continue

        # Group data by fairness f
        if f not in fairness:
            fairness[f] = {}
        if x not in fairness[f]:
            fairness[f][x] = []
        fairness[f][x].append(y)

    # pprint.pprint(fairness)
    # sys.exit(0)

    keys = list(fairness.keys())
    fairness_plt = {}
    for f in keys:
        fairness_plt[f] = {'x' : [], 'y' : []}
        for x in fairness[f]:
            median = statistics.median(fairness[f][x])
            fairness_plt[f]['x'].append(x)
            fairness_plt[f]['y'].append(median)

    # pprint.pprint(fairness_plt)
    # sys.exit(0)

    # plot
    keys = list(fairness_plt.keys())
    print(keys)
    keys.reverse()
    for f in keys:
        print(f)
        x = fairness_plt[f]['x']
        y = fairness_plt[f]['y']
        # ax.scatter(x, y, c=str(f), alpha = 0.5, label=f, cmap=cm.viridis)
        ax.scatter(x, y, c=str(f), label=f, cmap=cm.viridis)

    # x = []
    # y = []
    # for val in vals:
    #     if val['fairness-thresh'] == 0.8: #FAIRNESS_THRESH:
    #         x.append(val['num-input-points'])
    #         y.append(val['time-total'])
    # ax.scatter(x, y, c="red", label="Fairness=0.8") # close, s=volume, alpha=0.5)
    
    ax.set_xlabel(r'Fairness threshold', fontsize=15)
    ax.set_ylabel(r'Total time (s)', fontsize=15)
    ax.set_title('')
    
    # https://stackoverflow.com/questions/22239691/code-for-best-fit-straight-line-of-a-scatter-plot-in-python
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.polyfit.html
    # plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), linestyle="--", label="")
    # Returns a vector of coefficients p that minimises the squared error
    # plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), linestyle="--", label="Min square error 1D fit")
    # TODO this is better: http://seaborn.pydata.org/tutorial/regression.html#functions-to-draw-linear-regression-models

    # Set aspect ratio:
    # set_aspect_ratio(ax, 0.5)
    set_aspect_ratio(ax, 0.25)
    ax.grid(True)
    ax.legend(title="Fairness thresh")
    fig.tight_layout()
    return fig


def plot7(args,vals):
    '''
    x: number of data points in the dataset
    y: total runtime
    '''
    fig, ax = plt.subplots()
    

    fairness = {}
    for val in vals:
        f = val['fairness-thresh']
        x = val['alpha']
        y = val['time-total']

        # if f < 0.7:
        #     continue

        # Group data by fairness f
        if f not in fairness:
            fairness[f] = {}
        if x not in fairness[f]:
            fairness[f][x] = []
        fairness[f][x].append(y)

    # pprint.pprint(fairness)
    # sys.exit(0)

    keys = list(fairness.keys())
    fairness_plt = {}
    for f in keys:
        fairness_plt[f] = {'x' : [], 'y' : []}
        for x in fairness[f]:
            median = statistics.median(fairness[f][x])
            fairness_plt[f]['x'].append(x)
            fairness_plt[f]['y'].append(median)

    # pprint.pprint(fairness_plt)
    # sys.exit(0)

    # plot
    keys = list(fairness_plt.keys())
    print(keys)
    keys.reverse()
    for f in keys:
        print(f)
        x = fairness_plt[f]['x']
        y = fairness_plt[f]['y']
        # ax.scatter(x, y, c=str(f), alpha = 0.5, label=f, cmap=cm.viridis)
        ax.scatter(x, y, c=str(f), label=f, cmap=cm.viridis)

    # x = []
    # y = []
    # for val in vals:
    #     if val['fairness-thresh'] == 0.8: #FAIRNESS_THRESH:
    #         x.append(val['num-input-points'])
    #         y.append(val['time-total'])
    # ax.scatter(x, y, c="red", label="Fairness=0.8") # close, s=volume, alpha=0.5)
    
    ax.set_xlabel(r'Alpha', fontsize=15)
    ax.set_ylabel(r'Total time (s)', fontsize=15)
    ax.set_title('')
    
    # https://stackoverflow.com/questions/22239691/code-for-best-fit-straight-line-of-a-scatter-plot-in-python
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.polyfit.html
    # plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), linestyle="--", label="")
    # Returns a vector of coefficients p that minimises the squared error
    # plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), linestyle="--", label="Min square error 1D fit")
    # TODO this is better: http://seaborn.pydata.org/tutorial/regression.html#functions-to-draw-linear-regression-models

    # Set aspect ratio:
    # set_aspect_ratio(ax, 0.5)
    set_aspect_ratio(ax, 0.25)
    ax.grid(True)
    ax.legend(title="Fairness thresh")
    fig.tight_layout()
    return fig


def plot8(args,vals):
    '''
    x: number of paths in the input decisoin tree
    y: total runtime
    '''
    fig, ax = plt.subplots()
    fairness = {}
    for val in vals:
        f = val['fairness-thresh']
        x = val['forest-size']
        y = val['average-syntactic-change']
        # z = ['num-hcubes-split-all']

        # if f < 0.7:
        #     continue

        # Group data by fairness f
        if f not in fairness:
            fairness[f] = {}
        if x not in fairness[f]:
            fairness[f][x] = []
        fairness[f][x].append(y)

    # pprint.pprint(fairness)
    # sys.exit(0)

    keys = list(fairness.keys())
    fairness_plt = {}
    for f in keys:
        fairness_plt[f] = {'x' : [], 'y' : []}
        for x in fairness[f]:
            median = statistics.median(fairness[f][x])
            fairness_plt[f]['x'].append(x)
            fairness_plt[f]['y'].append(median)

    #pprint.pprint(fairness_plt)
    # sys.exit(0)

    # plot
    keys = list(fairness_plt.keys())
    print(keys)
    keys.reverse()
    for f in keys:
        print(f)
        x = fairness_plt[f]['x']
        y = fairness_plt[f]['y']
        # ax.scatter(x, y, c=str(f), alpha = 0.5, label=f, cmap=cm.viridis)
        ax.scatter(x, y, c=str(f), label=f, cmap=cm.viridis)

    ax.set_xlabel(r'# of trees in the forest', fontsize=15)
    ax.set_ylabel(r'Syntactic change', fontsize=15)
    ax.set_title('')

    # plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='City Area')
    ax.legend(title="Fairness thresh")

    # Set aspect ratio:
    set_aspect_ratio(ax, 0.5)

    ax.grid(True)
    fig.tight_layout()
    return fig


def plot9(args,vals):
    '''
    x: number of paths in the input decisoin tree
    y: total runtime
    '''
    fig, ax = plt.subplots()
    fairness = {}
    for val in vals:
        f = val['fairness-thresh']
        x = val['num-input-points']
        if 'average-syntactic-change' in val.keys():
            y = val['average-syntactic-change']

        # if f < 0.7:
        #     continue

        # Group data by fairness f
        if f not in fairness:
            fairness[f] = {}
        if x not in fairness[f]:
            fairness[f][x] = []
        fairness[f][x].append(y)

    # pprint.pprint(fairness)
    # sys.exit(0)

    keys = list(fairness.keys())
    fairness_plt = {}
    for f in keys:
        fairness_plt[f] = {'x' : [], 'y' : []}
        for x in fairness[f]:
            median = statistics.median(fairness[f][x])
            fairness_plt[f]['x'].append(x)
            fairness_plt[f]['y'].append(median)

    #pprint.pprint(fairness_plt)
    # sys.exit(0)

    # plot
    keys = list(fairness_plt.keys())
    print(keys)
    keys.reverse()
    for f in keys:
        print(f)
        x = fairness_plt[f]['x']
        y = fairness_plt[f]['y']
        # ax.scatter(x, y, c=str(f), alpha = 0.5, label=f, cmap=cm.viridis)
        ax.scatter(x, y, c=str(f), label=f, cmap=cm.viridis)

    ax.set_xlabel(r'# of points in dataset', fontsize=15)
    ax.set_ylabel(r'Syntactic change', fontsize=15)
    ax.set_title('')

    # plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='City Area')
    ax.legend(title="Fairness thresh")

    # Set aspect ratio:
    set_aspect_ratio(ax, 0.5)

    ax.grid(True)
    fig.tight_layout()
    return fig


def plot10(args,vals):
    '''
    x: number of paths in the input decisoin tree
    y: total runtime
    '''
    fig, ax = plt.subplots()
    fairness = {}
    for val in vals:
        f = val['fairness-thresh']
        x = val['fairness-thresh']
        y = val['average-syntactic-change']

        # if f < 0.7:
        #     continue

        # Group data by fairness f
        if f not in fairness:
            fairness[f] = {}
        if x not in fairness[f]:
            fairness[f][x] = []
        fairness[f][x].append(y)

    # pprint.pprint(fairness)
    # sys.exit(0)

    keys = list(fairness.keys())
    fairness_plt = {}
    for f in keys:
        fairness_plt[f] = {'x' : [], 'y' : []}
        for x in fairness[f]:
            median = statistics.median(fairness[f][x])
            fairness_plt[f]['x'].append(x)
            fairness_plt[f]['y'].append(median)

    #pprint.pprint(fairness_plt)
    # sys.exit(0)

    # plot
    keys = list(fairness_plt.keys())
    print(keys)
    keys.reverse()
    for f in keys:
        print(f)
        x = fairness_plt[f]['x']
        y = fairness_plt[f]['y']
        # ax.scatter(x, y, c=str(f), alpha = 0.5, label=f, cmap=cm.viridis)
        ax.scatter(x, y, c=str(f), label=f, cmap=cm.viridis)

    ax.set_xlabel(r'Fairness threshold', fontsize=15)
    ax.set_ylabel(r'Syntactic change', fontsize=15)
    ax.set_title('')

    # plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='City Area')
    ax.legend(title="Fairness thresh")

    # Set aspect ratio:
    set_aspect_ratio(ax, 0.5)

    ax.grid(True)
    fig.tight_layout()
    return fig


def plot11(args,vals):
    '''
    x: number of paths in the input decisoin tree
    y: total runtime
    '''
    fig, ax = plt.subplots()
    fairness = {}
    for val in vals:
        f = val['fairness-thresh']
        x = val['alpha']
        y = val['average-syntactic-change']

        # if f < 0.7:
        #     continue

        # Group data by fairness f
        if f not in fairness:
            fairness[f] = {}
        if x not in fairness[f]:
            fairness[f][x] = []
        fairness[f][x].append(y)

    # pprint.pprint(fairness)
    # sys.exit(0)

    keys = list(fairness.keys())
    fairness_plt = {}
    for f in keys:
        fairness_plt[f] = {'x' : [], 'y' : []}
        for x in fairness[f]:
            median = statistics.median(fairness[f][x])
            fairness_plt[f]['x'].append(x)
            fairness_plt[f]['y'].append(median)

    #pprint.pprint(fairness_plt)
    # sys.exit(0)

    # plot
    keys = list(fairness_plt.keys())
    print(keys)
    keys.reverse()
    for f in keys:
        print(f)
        x = fairness_plt[f]['x']
        y = fairness_plt[f]['y']
        # ax.scatter(x, y, c=str(f), alpha = 0.5, label=f, cmap=cm.viridis)
        ax.scatter(x, y, c=str(f), label=f, cmap=cm.viridis)

    ax.set_xlabel(r'Alpha', fontsize=15)
    ax.set_ylabel(r'Syntactic change', fontsize=15)
    ax.set_title('')

    # plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='City Area')
    ax.legend(title="Fairness thresh")

    # Set aspect ratio:
    set_aspect_ratio(ax, 0.5)

    ax.grid(True)
    fig.tight_layout()
    return fig


def plot12(args,vals):
    '''
    x: number of paths in the input decisoin tree
    y: total runtime
    '''
    fig, ax = plt.subplots()
    fairness = {}
    for val in vals:
        f = val['fairness-thresh']
        x = val['num-hcubes-split-all']
        x = sum(x)/len(x)
        if 'average-syntactic-change' in val.keys():
            y = val['average-syntactic-change']

        # if f < 0.7:
        #     continue

        # Group data by fairness f
        if f not in fairness:
            fairness[f] = {}
        if x not in fairness[f]:
            fairness[f][x] = []
        fairness[f][x].append(y)

    # pprint.pprint(fairness)
    # sys.exit(0)

    keys = list(fairness.keys())
    fairness_plt = {}
    for f in keys:
        fairness_plt[f] = {'x' : [], 'y' : []}
        for x in fairness[f]:
            median = statistics.median(fairness[f][x])
            fairness_plt[f]['x'].append(x)
            fairness_plt[f]['y'].append(median)

    #pprint.pprint(fairness_plt)
    # sys.exit(0)

    # plot
    keys = list(fairness_plt.keys())
    print(keys)
    keys.reverse()
    for f in keys:
        print(f)
        x = fairness_plt[f]['x']
        y = fairness_plt[f]['y']
        # ax.scatter(x, y, c=str(f), alpha = 0.5, label=f, cmap=cm.viridis)
        ax.scatter(x, y, c=str(f), label=f, cmap=cm.viridis)
    
    ax.plot(x, [x_*0.05 for x_ in x], label="5%")

    vals = ax.get_yticks()
    # ax.set_yticklabels(['{:,.2%}'.format(x_/488.42) for x_ in y])
    ax.set_xlabel('Average initial number of path hypercubes \nfor random forests trained on the Adult dataset', fontsize=15)
    ax.set_ylabel('Syntactic change', fontsize=15)
    ax.set_title('')

    # plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='City Area')
    ax.legend(title="Fairness thresh")

    # Set aspect ratio:
    set_aspect_ratio(ax, 0.5)

    ax.grid(True)
    fig.tight_layout()
    return fig


def plot13(args,vals):
    '''
    x: number of paths in the input decisoin tree
    y: total runtime
    '''
    fig, ax = plt.subplots()
    m = []
    n = []

    for val in vals:
        x = val['num-hcubes-split-all']
        x = sum(x)/len(x)
        if 'average-syntactic-change' in val.keys():
            y = val['average-syntactic-change']
        m.append(x)
        n.append(y)

    ax.scatter(m, n,color='grey')
    
    ax.plot(m, [x_*0.05 for x_ in m], label="y = 0.05x")

    vals = ax.get_yticks()
    # ax.set_yticklabels(['{:,.2%}'.format(x_/488.42) for x_ in y])
    ax.set_xlabel('Average initial number of path hypercubes in\ndecision trees for random forests (adult dataset)', fontsize=15)
    ax.set_ylabel('Syntactic change', fontsize=15)
    ax.set_title('')

    # plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='City Area')
    ax.legend()

    # Set aspect ratio:
    set_aspect_ratio(ax, 0.25)

    ax.grid(True)
    fig.tight_layout()
    return fig


def plot14(args,vals):
    f1 = {}
    f2 = {}
    f3 = {}
    f4 = {}
    f5 = {}
    f6 = {}
    for i in [0.5,0.6,0.7,0.8,0.9,0.95]:
        f1[i] = []
        f2[i] = []
        f3[i] = []
        f4[i] = []
        f5[i] = []
        f6[i] = []
    for val in vals:
        f1[val['fairness-thresh']].append(val['accuracy-before']*100)
        f2[val['fairness-thresh']].append(val['accuracy-after']*100)
        f3[val['fairness-thresh']].append(val['precision-before']*100)
        f4[val['fairness-thresh']].append(val['precision-after']*100)
        f5[val['fairness-thresh']].append(val['recall-before']*100)
        f6[val['fairness-thresh']].append(val['recall-after']*100)

    g1 = {}
    g2 = {}
    g3 = {}
    g4 = {}
    g5 = {}
    g6 = {}
    for i in [0.5,0.6,0.7,0.8,0.9,0.95]:
        g1[i] = statistics.mean(f1[i])
        g2[i] = statistics.mean(f2[i])
        g3[i] = statistics.mean(f3[i])
        g4[i] = statistics.mean(f4[i])
        g5[i] = statistics.mean(f5[i])
        g6[i] = statistics.mean(f6[i])
    
    x = [0.5,0.6,0.7,0.8,0.9,0.95]
    y1 = list(g1.values())
    y2 = list(g2.values())
    y3 = list(g3.values())
    y4 = list(g4.values())
    y5 = list(g5.values())
    y6 = list(g6.values())
    fig = plt.figure()
    fig.tight_layout()
    ax = fig.add_subplot(111)
    print(y1)
    print(y2)
    ax.plot(x,y1,label="Adult (forest) accuracy before",marker="o",ls="--")
    ax.plot(x,y2,label="Adult (forest) accuracy after",marker="o")
    # ax.plot(x,y3,label="Adult (forest) precision before",marker="o")
    # ax.plot(x,y4,label="Adult (forest) precision after",marker="o")
    # ax.plot(x,y5,label="Adult (forest) recall before",marker="o")
    # ax.plot(x,y6,label="Adult (forest) recall after",marker="o")
    plt.ylim(80,100)
    ax.legend()
    ax.grid(True)
    ax.set_ylabel("Accuracy (%)",fontsize=12)
    ax.set_xlabel("Fairness Threshold",fontsize=12)
    ax.set_title("Classification Accuracy Rate v.s. Fairness Threshold",fontsize=15)
    set_aspect_ratio(ax, 0.3)
    
    # fig.savefig('./accuracy.png')
    return fig


def main(args, vals):
    pprint.pprint(args)
    # pprint.pprint(vals)

    # Figure out which plot type was selected
    if args.plot_type == 1:
        f = plot1(args,vals)
    elif args.plot_type == 2:
        f = plot2(args,vals)
    elif args.plot_type == 3:
        f = plot3(args,vals)
    elif args.plot_type == 4:
        f = plot4(args,vals)
    elif args.plot_type == 5:
        f = plot5(args,vals)
    elif args.plot_type == 6:
        f = plot6(args,vals)
    elif args.plot_type == 7:
        f = plot7(args,vals)
    elif args.plot_type == 8:
        f = plot8(args,vals)
    elif args.plot_type == 9:
        f = plot9(args,vals)
    elif args.plot_type == 10:
        f = plot10(args,vals)
    elif args.plot_type == 11:
        f = plot11(args,vals)
    elif args.plot_type == 12:
        f = plot12(args,vals)
    elif args.plot_type == 13:
        f = plot13(args,vals)
    elif args.plot_type == 14:
        f = plot14(args,vals)
    # Save the plot to file, if any was specified
    if args.plot_output:
        print("Saving plot to file:", args.plot_output)
        f.savefig(args.plot_output, bbox_inches='tight')

    # Show the plot
    # plt.show()
    return


def parse_args():
    parser = argparse.ArgumentParser(description='Plots data from pickled evaluation file(s)')
    parser.add_argument('-t', '--plot-type', type=int, help='Type of plot to generate',
                        required=True)
    parser.add_argument('-o', '--plot-output', type=str, help='Where to store plot output',
                        required=False)
    parser.add_argument('eval_files',
                        nargs='+', 
                        type=str,
                        help='(multiple) filenames with pickled evaluation data')
    args = parser.parse_args()

    # Restore the pickled file(s) into a *list of dictionaries*
    vals = []
    for f in args.eval_files:
        val = EvalUtil.restore_vals(f)
        vals.append(val)
    return args, vals


if __name__ == '__main__':
    args, vals = parse_args()
    main(args, vals)
