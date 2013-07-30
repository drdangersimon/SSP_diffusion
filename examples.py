'''Performs examples of the diffuson mapping'''

from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
import numpy as nu
import pylab as lab
from mpl_toolkits.mplot3d import Axes3D as lab3d
from scipy.cluster.vq import kmeans2
import diffuse as dif

def diffusion_example(files='annulus.mat',test_flag=True,flag_t=False):
    Data = loadmat(files)['Data']
    n,p = Data.shape  # Data(n,p), where n=#observations, p=#variables
    lab.plot(Data[:,0],Data[:,1],'.') #axis image
    lab.title('Example data');
    #lab.show()
    D = squareform(pdist(Data)) # pairwise distances, n-by-n matrix

    #--------------------------------------------------------------------
    # CHECK THE DISTRIBUTION OF NEAREST NEIGHBORS
    #--------------------------------------------------------------------
    
    if test_flag: # Estimate distribution of k-nearest neighbor
        D_sort = nu.sort(D,1)
        k = 30 #30
        dist_knn = D_sort[:,1+k]  # distance to k-nn
        median_val = nu.median(dist_knn)
        eps_val = median_val**2/2.
        sigmaK = nu.sqrt(2) * median_val
        lab.figure()
        lab.hist(dist_knn)
        lab.title('Distribution of distance to k-NN')


    #--------------------------------------------------------------------
    # SET PARAMETERS IN MODEL
    #--------------------------------------------------------------------
    eps_val = 0.05  
    neigen = 10
    #flag_t =  #flag_t=0 => Default: multi-scale geometry
    if flag_t:
        t=3  # fixed time scale  


    #--------------------------------------------------------------------
    # EIGENDECOMPOSITION
    #--------------------------------------------------------------------
    # Call function "diffuse.diffuse"
    X, eigenvals, psi, phi = dif.diffuse(D,eps_val,neigen)


    #--------------------------------------------------------------------
    # DIFFUSION K-MEANS
    #--------------------------------------------------------------------
    k=2  # number of clusters
    #go till centroids don't change
    cen,i = None,0
    while True:
        if cen is None:
            cen, lables = kmeans2(X, k)
            i += 1
            continue
        else:
            t_cen, lables = kmeans2(X, cen,iter=300, minit='matrix')
        #check is centroids have changed
        if (nu.allclose(cen,t_cen) and i > 4) or i > 100:
            break
        else:
            i += 1
            cen = t_cen.copy()

    #--------------------------------------------------------------------
    # PLOT RESULTS
    #--------------------------------------------------------------------
    fig1 = lab.figure() # Fall-off of eigenvalues
    plt1 = fig1.add_subplot(211)
    plt1.plot(eigenvals[1:neigen+1],'*-') # non-trivial eigenvals
    plt1.set_ylabel('Eigenvalues \lambda');
    if flag_t:
        lambda_t = eigenval[1:neigen+1]**t 
        plt2 = fig1.add_subplot(212)
        plt2.plot(lambda_t,'*-')  # multiscale weighting
        plt2.set_ylabel('\lambda^t')
    else:
        lambda_multi = eigenvals[1:neigen+1] / (1-eigenvals[1:neigen+1])
        plt2 = fig1.add_subplot(212)
        plt2.plot(lambda_multi,'*-')  # multiscale weighting
        plt2.set_ylabel('\lambda / (1-\lambda)');

    fig2 = lab.figure() # Diffusion map
    plt3d = fig2.add_subplot(111, projection='3d')
    plt3d.scatter3D(X[:,0],X[:,1],X[:,2],s=10,c='b') 
    plt3d.set_title('Embedding with first 3 diffusion coordinates')
    plt3d.set_xlabel('X_1')
    plt3d.set_ylabel('X_2')
    plt3d.set_zlabel('X_3')
    
    plt3d.scatter3D(cen[:,0],cen[:,1],cen[:,2],s=60,c='k') 

    fig3 = lab.figure() # K-means labeling
    plt3 = fig3.add_subplot(111)
    plt3.scatter(Data[:,0],Data[:,1],20,c=lables)
    plt3.set_title('K-means with K=2')
    lab.show()


def ssp_example(flag_t=False):
    
   
    ssps = nu.loadtxt('ssps.txt') # load SSP spectra
    params = nu.loadtxt('ssp_params.txt') # load SSP age/Z
    t = params[:,0] # SSP ages
    Z = params[:,1] # SSP Zs
    lam = nu.loadtxt('ssps_wl.txt') # load wavelength dispersion
    n,p = ssps.shape # Data(n,p), where n=#observations, p=#variables
    lam0 = nu.where(lam==4020)[0] # normalization wavelength
    norm = ssps[:,lam0].copy() # keep track of normalization const.
    ssps = ssps / nu.tile(norm,(1,p)) # normalize all SSPs at lam0

    D = squareform(pdist(ssps)) # pairwise distances, n-by-n matrix'''

    #--------------------------------------------------------------------
    # SET PARAMETERS IN MODEL
    #--------------------------------------------------------------------
    eps_val = 500/ 4.
    if flag_t
        t=3 # fixed time scale  
        

    #--------------------------------------------------------------------
    # EIGENDECOMPOSITION
    #--------------------------------------------------------------------
    # Call function "diffuse.m"
    print('Computing diffuson coefficients')
    X, eigenvals, psi, phi = dif.diffuse(D,eps_val)


    #--------------------------------------------------------------------
    # DIFFUSION K-MEANS
    #--------------------------------------------------------------------
    print('Running diffusion K-means')
    k = 45  # number of clusters
    cen ,i = None, 0
    while True:
        if cen is None:
            cen, lables = kmeans2(X, k)
            i += 1
            continue
        else:
            t_cen, lables = kmeans2(X, cen,iter=30, minit='matrix')
        #check is centroids have changed
        if (nu.allclose(cen,t_cen) and i > 4) or i > 100:
            break
        else:
            i += 1
            cen = t_cen.copy()

    protospecnorm = nu.zeros((k, ssps.shape[1]))
    protoages, protometals, norms = nu.zeros(k), nu.zeros(k),  nu.zeros(k)
    for jj in range(k): # define prototype parameters
        protospecnorm[jj,:]=nu.sum(nu.tile(phi0[lables==jj]/nu.sum(phi0[lables==jj]),(p,1)).T
                                   * ssps[lables==jj,:],0)
        protoages[jj] = nu.sum(phi0[lables==jj]/nu.sum(phi0[lables==jj]) * t[lables==jj],0)
        protometals[jj] = nu.sum(phi0[lables==jj]/nu.sum(phi0[lables==jj]) * Z[lables==jj],0)
        norms[jj] = nu.sum(phi0[lables==jj]/nu.sum(phi0[lables==jj]) *norm[lables==jj])

    protospec= protospecnorm * nu.tile(norms,(p,1)).T

    #--------------------------------------------------------------------
    # PLOT RESULTS
    #--------------------------------------------------------------------
    '''figure, % 3-dimensional diffusion map of 1278 SSPs
    scatter3(X(:,1),X(:,2),X(:,3),20,idx,'filled'); 
    title('Embedding of SSPs with first 3 diffusion coordinates plus K-means centroids');
    xlabel('X_1'); ylabel('X_2'); zlabel('X_3');
    hold on
    scatter3(C(:,1),C(:,2),C(:,3),45,'k','filled'); '''


    # Cid Fernandes et al.(2005) parameters
    tCF = nu.asarray([0.001, 0.00316, 0.00501 ,0.01 ,0.02512 ,0.04 ,0.10152 ,0.28612 ,0.64054, 0.90479 ,1.434 ,2.5 ,5 ,11 ,13])
    ZCF = nu.asarray([0.0040 , 0.0200,  0.0500])
    idxCF=[] # find SSP spectra used by Cid Fernandes et al.(2005)
    for ii in range(15):
        for jj in range(3):
            idxCF.append(nu.logical_and(nu.where(t==tCF[ii])[0] ,nu.where(Z == ZCF[jj])[0]))

% plot normalized prototype spectra, colored by log age
cmap = colormap;
CFcolor = ceil((log10(t(idxCF))-min(log10(protoages)))/(max(log10(protoages))-min(log10(protoages)))*63)+1;
dmapcolor = ceil((log10(protoages)-min(log10(protoages)))/(max(log10(protoages))-min(log10(protoages)))*63)+1;

figure, % prototype spectra, colored by log age
subplot(2,1,1)
for ii=1:45,
  plot(lam,ssps(idxCF(ii),:),'Color',cmap(CFcolor(ii),:),'LineWidth',.01)
  hold on
end
xlabel('lambda'); ylabel('Normalized flux');ylim([0 3]);
title('Normalized Cid Fernandes et al.(2005) prototype spectra, K=45, colored by log age');
subplot(2,1,2)
for ii=1:k,
  plot(lam,protospecnorm(ii,:),'Color',cmap(dmapcolor(ii),:),'LineWidth',.01)
  hold on
end
xlabel('lambda'); ylabel('Normalized flux'); ylim([0 3]);
title(char(strcat('Normalized diffusion map prototype spectra, K=',num2str(k),', colored by log age')));


% plot normalized prototype spectra, colored by log Z
CFcolor = ceil((log10(Z(idxCF))-min(log10(protometals)))/(max(log10(protometals))-min(log10(protometals)))*63)+1;
dmapcolor = ceil((log10(protometals)-min(log10(protometals)))/(max(log10(protometals))-min(log10(protometals)))*63)+1;

figure, % prototype spectra, colored by log Z
subplot(2,1,1)
for ii=1:45,
  plot(lam,ssps(idxCF(ii),:),'Color',cmap(CFcolor(ii),:),'LineWidth',.01)
  hold on
end
xlabel('lambda'); ylabel('Normalized flux');ylim([0 3]);
title('Normalized Cid Fernandes et al.(2005) prototype spectra, K=45, colored by log Z');
subplot(2,1,2)
for ii=1:k,
  plot(lam,protospecnorm(ii,:),'Color',cmap(dmapcolor(ii),:),'LineWidth',.01)
  hold on
end
xlabel('lambda'); ylabel('Normalized flux'); ylim([0 3]);
title(char(strcat('Normalized diffusion map prototype spectra, K=',num2str(k),', colored by log Z')));

'''
if __name__ == '__main__':

     diffusion_example()
