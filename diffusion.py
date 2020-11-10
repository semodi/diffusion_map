import numpy as np
from sklearn.gaussian_process.kernels import RBF
import numpy as np
from sklearn.gaussian_process.kernels import RBF
import pandas as pd
import matplotlib.pyplot as plt
from scipy import misc,ndimage

def diffusion(X, epsilon = 0.0003):
    
    k = RBF(length_scale=np.sqrt(X.shape[1]*epsilon))

    kern = k(X,X)

    zl=np.sqrt(np.sum(kern,axis=1,keepdims=True))

    kern = kern/np.outer(np.sqrt(np.sum(kern,axis=0)),np.sqrt(np.sum(kern,axis=1)))


    eig = np.linalg.eigh(kern)
    eig = [eig[0][::-1].real, eig[1][:,::-1].round(4).real]
    
    eig[1] = eig[1]/zl
    return eig, kern  


def get_spin_data(n_classes=3):
    N = 64
    m = 1000
    def get_data(nu):
        theta_bar = np.random.rand(m)*2*np.pi
        theta_fluc = (np.random.randn(N*m).reshape(m,N))*np.pi/5
        theta =  np.arange(N)/N*2*np.pi*nu 
        theta = theta.reshape(1,-1) + theta_bar.reshape(-1,1) + theta_fluc
        X = np.concatenate([np.cos(theta),np.sin(theta)],axis=-1)
        return X

    X = np.concatenate([get_data(i) for i in range(n_classes)])
    y = np.concatenate([np.ones(m)*i for i in range(n_classes)])
    
    return X, y, n_classes

def get_si_data(gaussian=0, return_raw=True,normalize=False):
    X = np.concatenate([np.load('../unfolding/zijian/data2/0vacancy.npy'),
                       np.load('../unfolding/zijian/data2/1vacancy.npy'),
                       np.load('../unfolding/zijian/data2/2vacancy.npy')])
    y = np.concatenate([np.zeros(len(np.load('../unfolding/zijian/data2/0vacancy.npy'))),
                       np.ones(len(np.load('../unfolding/zijian/data2/1vacancy.npy'))),
                       2*np.ones(len(np.load('../unfolding/zijian/data2/2vacancy.npy')))])

    if gaussian:
        X = np.array([ndimage.gaussian_filter(x, sigma=gaussian) for x in X]) 
        
    X_raw = np.array(X)
    X = X.reshape(len(X),-1)

    feat_filt = (np.std(X,axis=0) > 1e-10)
    X = X[:,feat_filt]
    if normalize:
        X /= np.sum(X,axis=-1).reshape(-1,1)
    if return_raw:
         return X, y, len(np.unique(y)), X_raw
       
    return X, y, len(np.unique(y))

def get_graphene_data(lim=None):

    X = np.load('../unfolding/yue/Code/X_data_final_1.npy')
    y = np.load('../unfolding/yue/Code/y_data_final_1.npy')[:]

    if lim:
        X = X.reshape(len(X),-1)[:lim]
        y = y[:lim]

    feat_filt = (np.std(X,axis=0) > 1e-10)
    X = X[:,feat_filt]
    return X, y, len(np.unique(y))

def plot_summary(eig, y, n_classes, plotly=False):

    plt.figure()
    plt.title('Eigenvalues')
    plt.plot(eig[0][:10],ls='',marker='.')
    plt.ylabel('EV')
    plt.xlabel('idx')
    plt.grid()
    plt.figure()
    plt.title('Lower dimensional embedding')
    for i in range(0,n_classes):
        x = eig[1][y==i,1]
        z = eig[1][y==i,2]
        plt.plot(x, z, ls='',marker='o', label='Class {}'.format(i))
        plt.xlabel('$\psi_1$')
        plt.ylabel('$\psi_2$')
    plt.legend()
    
    if plotly:
        df = pd.DataFrame(eig[1][:,1:4])
        df.columns = ['EV1','EV2','EV3']
        df['Class'] = y.astype(int).astype(str)
        df['idx'] = np.array(df.index).astype(int)
        # df = df[df.idx < 200]


        import plotly.express as px

        fig = px.scatter_3d(df, x='EV1',y='EV2',z='EV3',color='Class',hover_data=['idx'],opacity=0.9)


        # tight layout
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        return fig

# Silicon
