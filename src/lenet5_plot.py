# .py file to draw 4 kinds of plots in paper for LeNet5 Experiments ... 
import os 
import pandas as pd
import matplotlib.pyplot as plt

"""
We may need moving average to make plots look better ... 
"""

path = "../results/"       # add your path to access your result files ...
sheets = [ "heterogeneous", "constant", "randomized_gossip", "zero" ]
names = [ 'EF-HC', 'GT', 'Random Gossip', 'ZT' ]
colors = ['b', 'r', 'g', 'black']

def plot_1( ):
    # file = "LeNet5_FMNIST_3epochs_10m_0.4conn_0.8weak_non_iid_Nonelabels_50.0r_conns_iter.xlsx"
    file = "LeNet5_FMNIST_3epochs_10m_0.4conn_0.8weak_labels_per_agent_1labels_50.0r_labels_iter.xlsx"
    fig, ax = plt.subplots()
    leng = 4
    for i in range(4):
        # we need 2 cols - 1) iters_sampled, 2) accuracies
        data = pd.read_excel( path+file, sheet_name=sheets[i] ) 
        ys = list( data['transmission_time_useds'] [:10000] )
        ys = [ sum( ys[i*leng:(i+1)*leng] ) / leng for i in range( int(len(ys)/leng) ) ] 
        ys = ys[::10]
        print( names[i], ys[:10] )
        plt.plot( [ x*leng*10 for x in range(len(ys))], ys, colors[i], label=names[i])

    ax.set_xlim([0, 10000])
    ax.set_ylim([0,52])

    ax.set_xlabel('iterations')
    ax.set_ylabel('transmission_time_units')
    fig.suptitle('i')   
    plt.legend()
    plt.savefig( path + '/lenet5_fig1.jpg') 


def plot_2( ):
    # file =  "LeNet5_FMNIST_3epochs_10m_0.4conn_0.8weak_non_iid_Nonelabels_50.0r_conns_iter_sampled.xlsx"
    file = "LeNet5_FMNIST_3epochs_10m_0.4conn_0.8weak_labels_per_agent_1labels_50.0r_labels_iter_sampled.xlsx"
    fig, ax = plt.subplots()
    for i in range(4):
        # we need 2 cols - 1) iters_sampled, 2) accuracies
        data = pd.read_excel( path + file, sheet_name=sheets[i] )
        xs, ys = data['iters_sampled'], data['accuracies']
        plt.plot( xs,ys, colors[i], label=names[i])

    ax.set_xlim([0, 11000])
    ax.set_ylim([0.1, 0.8])

    ax.set_xlabel('iterations')
    ax.set_ylabel('accuracies')
    fig.suptitle('ii')   
    plt.legend()
    plt.savefig( path + '/lenet5_fig2.jpg' )   


def plot_3( ):
    # need to aggregate transmission time units and accuracies from two consistent files by iterations values
    # file_a =  "LeNet5_FMNIST_3epochs_10m_0.4conn_0.8weak_non_iid_Nonelabels_50.0r_conns_iter_sampled.xlsx"
    # file_t = "LeNet5_FMNIST_3epochs_10m_0.4conn_0.8weak_non_iid_Nonelabels_50.0r_conns_iter.xlsx"
    file_a = "LeNet5_FMNIST_3epochs_10m_0.4conn_0.8weak_labels_per_agent_1labels_50.0r_labels_iter_sampled.xlsx"
    file_t = "LeNet5_FMNIST_3epochs_10m_0.4conn_0.8weak_labels_per_agent_1labels_50.0r_labels_iter.xlsx"
    fig, ax = plt.subplots()
    for i in range(4):
        dt = pd.read_excel( path+file_t, sheet_name=sheets[i] )
        da = pd.read_excel( path+file_a, sheet_name=sheets[i] )
        yst = list( dt['transmission_time_useds_cumsum'] [:100000] )
        xs, ysa = da['iters_sampled'], da['accuracies'] 
        xs = [ x for x in xs if x < 100000 ]
        ysa = ysa[:len(xs)] 
        yt = [ int(yst[x]) for x in xs ]
        plt.plot( yt, ysa, colors[i], label=names[i] ) 

    ax.set_xlim([0, 100000]) 
    ax.set_ylim([0.1, 0.8])

    ax.set_xlabel('transmission_time_units')
    ax.set_ylabel('accuracies')
    fig.suptitle('iii')   
    plt.legend()
    plt.savefig( path + '/lenet5_fig3.jpg' ) 

def plot_4( ):
    # different connection rate (keep other settings the same ...)
    # 0.2, 0.4, 0.6, 0.8, 1.0
    # manually copy and paste numbers from saved results 
    conns = [ [0.27122, 0.26043, 0.24603, 0.30198 ], [0.29432, 0.29042, 0.23727, 0.29671 ], [0.36348, 0.33663, 0.26006, 0.34102 ], [0.48247, 0.46273, 0.32382, 0.46313], [0.56052, 0.55917, 0.34989, 0.55642] ]
    xs = [ 0.2*i for i in range(1, len(conns)+1) ]
    fig, ax = plt.subplots()
    for i in range(4):
        accs = [ cs[i] for cs in conns ]
        plt.plot( xs, accs, colors[i], label=names[i] ) 

    ax.set_xlim([0.15, 1.05]) 
    ax.set_ylim([0.1, 0.6])

    ax.set_xlabel('graph connectivity')
    ax.set_ylabel('accuracy after 25000 total transmission time units')
    fig.suptitle('iv')   
    plt.legend()
    plt.savefig( path + '/lenet5_fig4.jpg') 


# def plot_reference( h,c,r,z ):
#     print( len(h), len(c), len(r), len(z) )
#     xs = [i for i in range(50)]
#     fig, ax = plt.subplots()
#     ax.plot([i for i in range(len(h)+1)], [0]+h, 'b', label = 'heterogeneous')
#     ax.plot([i for i in range(len(r)+1)], [0]+r, 'r', label = 'random gossip')
#     # ax.plot(epis, [0]+z, 'g', label = 'random_ratios')    # random_ratios
#
#     ax.set_xlim([0, 701])
#     ax.set_ylim([0, 1])
#
#     ax.set_xlabel('number of epochs')
#     ax.set_ylabel('accuracy')
#     fig.suptitle('')
#     plt.legend()
#     plt.savefig('fig2.jpg')


if __name__ == "__main__":
    plot_1()
    plot_2() 
    plot_3() 
    plot_4()