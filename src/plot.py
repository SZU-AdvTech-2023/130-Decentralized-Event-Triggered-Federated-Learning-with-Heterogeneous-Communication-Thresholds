# .py file to draw 4 kinds of plots in paper for SVM experiments ... 
import os 
import pandas as pd
import matplotlib.pyplot as plt

"""
We may need moving average to make plots look better ... 
"""

path = "../results/"           # add your path to access your result files ...
sheets = [ "heterogeneous", "constant", "randomized_gossip", "zero" ]
names = [ 'EF-HC', 'GT', 'Random Gossip', 'ZT' ]
colors = ['b', 'r', 'g', 'k']

def plot_1( ):
    # file = "SVM_FMNIST_10epochs_10m_0.4conn_0.8weak_non_iid_Nonelabels_50.0r_conns_iter.xlsx"
    file = "SVM_FMNIST_3epochs_10m_0.4conn_0.8weak_labels_per_agent_1labels_50.0r_labels_iter.xlsx"
    fig, ax = plt.subplots()
    leng = 4
    for i in range(4):
        # we need 2 cols - 1) iters_sampled, 2) accuracies
        data = pd.read_excel( path+file, sheet_name=sheets[i] ) 
        ys = list( data['transmission_time_useds'] [:20000] )
        ys = [ sum( ys[i*leng:(i+1)*leng] ) / leng for i in range( int(len(ys)/leng) ) ] 
        ys = ys[::10]
        print( names[i], ys[:10] )
        plt.plot( [ x*leng*10 for x in range(len(ys))], ys, colors[i], label=names[i])

    ax.set_xlim([0, 10000])
    ax.set_ylim([0,7.5])

    ax.set_xlabel('iterations')
    ax.set_ylabel('transmission_time_units')
    fig.suptitle('i')   
    plt.legend()
    plt.savefig(path + '/fig1_1label.jpg')


def plot_2( ):
    # file = "SVM_FMNIST_10epochs_10m_0.4conn_0.8weak_non_iid_Nonelabels_50.0r_conns_iter_sampled.xlsx"
    file =  "SVM_FMNIST_3epochs_10m_0.4conn_0.8weak_labels_per_agent_1labels_50.0r_labels_iter_sampled.xlsx"
    fig, ax = plt.subplots()
    for i in range(4):
        # we need 2 cols - 1) iters_sampled, 2) accuracies
        # if i == 1:
        #     data = pd.read_excel( path + file_gt, sheet_name=sheets[i] )
        # else:
        data = pd.read_excel( path + file, sheet_name=sheets[i] )
        xs, ys = data['iters_sampled'], data['accuracies']
        plt.plot( xs,ys, colors[i], label=names[i])

    ax.set_xlim([0, 10000])
    ax.set_ylim([0.4, 0.85])

    ax.set_xlabel('iterations')
    ax.set_ylabel('accuracies')
    fig.suptitle('ii')   
    plt.legend()
    plt.savefig(path + '/fig2_1label.jpg')


def plot_3( ):
    file_a = "SVM_FMNIST_3epochs_10m_0.4conn_0.8weak_labels_per_agent_1labels_50.0r_labels_iter_sampled.xlsx"
    file_t = "SVM_FMNIST_3epochs_10m_0.4conn_0.8weak_labels_per_agent_1labels_50.0r_labels_iter.xlsx"
    fig, ax = plt.subplots()
    for i in range(4):
        dt = pd.read_excel( path+file_t, sheet_name=sheets[i] )
        da = pd.read_excel( path+file_a, sheet_name=sheets[i] )
        yst = list( dt['transmission_time_useds_cumsum'] [:10000] )
        xs, ysa = da['iters_sampled'], da['accuracies'] 
        xs = [ x for x in xs if x < 10000 ]
        ysa = ysa[:len(xs)] 
        yt = [ int(yst[x]) for x in xs ]
        plt.plot( yt, ysa, colors[i], label=names[i] ) 

    ax.set_xlim([0, 10000]) 
    ax.set_ylim([0.45, 0.85])

    ax.set_xlabel('transmission_time_units')
    ax.set_ylabel('accuracies')
    fig.suptitle('iii')   
    plt.legend()
    plt.savefig(path + '/fig3_1label.jpg')

def plot_4( ):
    # different connection rate (keep other settings the same ...)
    # 0.2, 0.4, 0.6, 0.8, 1.0
    # manually copy and paste numbers from saved results 
    # file = 
    conns = [ [0.29228, 0.18504, 0.18648,0.28221], [0.29228, 0.29048, 0.19816, 0.28221], [0.37989, 0.36504, 0.28377, 0.51127], [0.48982, 0.41452, 0.27496, 0.48509], [0.55426, 0.49882, 0.30819, 0.51026] ]
    # conns = [ [0.21651, 0.19285, 0.20642,0.21651], [0.29228, 0.29048, 0.19816, 0.28221], [0.37989, 0.36504, 0.28377, 0.51127], [0.48982, 0.41452, 0.27496, 0.48509], [0.55426, 0.49882, 0.30819, 0.51026] ]
    xs = [ 0.2*i for i in range(1,6) ]
    fig, ax = plt.subplots()
    for i in range(4):
        accs = [ cs[i] for cs in conns ]
        plt.plot( xs, accs, colors[i]+"-o", label=names[i] ) 

    ax.set_xlim([0.15, 1.05]) 
    ax.set_ylim([0.1, 0.6])

    ax.set_xlabel('graph connectivity')
    ax.set_ylabel('accuracy after 15 total transmission time units')
    fig.suptitle('iv')   
    plt.legend()
    plt.savefig(path + '/fig4_1label.jpg')   


if __name__ == "__main__":
    plot_1()
    plot_2()
    plot_3()
    plot_4()
