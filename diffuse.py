'''Diffusion mapping by Ann B Lee, March 2009. Last modified, JWR: 3/23/2009 ported to python'''



import numpy as nu
from scipy.cluster.vq import kmeans2
#from scipy.sparse.linalg import svds as svd
from scipy.linalg import svd

def diffuse(D,eps_val,neigen=None,t=None,threshold = 5E-6):
    ''' diffuse(D,eps_val,t,neigen)
    (ndarray or matrix, float, float, float
    D          --- pairwise distances, n-by-n matrix
    eps_val    --- parameter in weight matrix exp(-D^2/(4*eps_val))
    neigen     --- number of dimensions of final representation
    (default: 95% drop-off in eigenvalue multiplier)
    t          --- optional time parameter in diffusion map 
    (default: model with multiscale geometry)    
    threshold  --- the value in D which should be ignored
    
    Output: 
    X          --- non-trivial diffusion coords. (entered column-wise)  
    eigenvals  --- eigenvalues of Markov matrix
    psi        --- right eigenvectors of Markov matrix
    phi        --- left eigenvectors'''

    assert D.shape[0] == D.shape[1]
    n = D.shape[0]
    K = nu.exp(-D**2/(4.*eps_val)) #or equivalently, K=exp(-(D/sigmaK).^2)
    v = nu.mat(nu.sqrt(nu.sum(K,1)))
    #v = v(:)
    A = K / nu.asarray(v.T * v)   # symmetric graph Laplacian
    
    #A[threshold < A] = 0
    #A = sparse(A)  # make matrix sparse to speed up calcs
    #print A.nnz
    if neigen is None:
        # eigendecomposition of symmetric matrix
        U,eigenvals,V = svd(A) #,51)
        first_colum = nu.tile(U[:,0],(51,1)).T
        eigenvals = eigenvals[:51]
        # right eigenv of Markov matrix
        psi = U[:,:51] / first_colum
        # left eigenv of Markov matrix
        phi = U[:,:51] * first_colum
        
    else:
        # eigendecomposition of symmetric matrix
        U,eigenvals,V = svd(A) #, neigen+1) 
        first_colum = nu.tile(U[:,0],(neigen+1,1)).T
        eigenvals = eigenvals[:neigen+1]
        # right eigenv of Markov matrix
        psi = U[:,:neigen+1] / first_colum
        # left eigenv of Markov matrix
        phi = U[:,:neigen+1] * first_colum

    #eigenvals.sort()
    # DIFFUSION COORDINATES
    if not t is None: # fixed scale parameter
        lambda_t = eigenvals[1:]**t
        lambda_t = nu.tile(lambda_t,(n,1))
        if neigen is None: # use neigen corresponding to 95% drop-off in lambda_t
            lam = lambda_t[0,:] / lambda_t[0,0]
            neigen = nu.sum(lam < .05) # default number of eigenvalues
            if neigen < 1:
                neigen = 50 # use neigen=50 if 95% dropoff not attained
            print 'Used default value: % dimensions'%neigen
  
        X = psi[:,1:neigen+1] * lambda_t[:,:neigen+1]  # diffusion coords X
        #= right eigenvectors rescaled with eigenvalues
    else:  # multiscale geomtry
        lambda_multi = eigenvals[1:] / (1-eigenvals[1:])
        lambda_multi = nu.tile(lambda_multi,(n,1))
        if neigen is None:  # use neigen corresponding to 95% drop-off in lambda_multi
            lam = lambda_multi[0,:] / lambda_multi[0,0]
            neigen = nu.min(lam[lam<.05])
            if neigen < 1:# default number of eigenvalues
                neigen =  50 # use neigen=50 if 95% dropoff not attained
            print 'Used default value: %i dimensions'%neigen
        X = psi[:,1:neigen+1] * lambda_multi[:,:neigen+1]  # diffusion coords X
        

    return X,eigenvals, psi, phi

def diffusion_kmeans(X, k, phi0, Niter, epsilon=10**-3):
    '''DIFFUSION_KMEANS diffusion K-means clustering
     diffusion_kmeans(X, k, phi0, Niter, epsilon)
     
     
    Input:
    X(n,d)     --- diffusion coords (right eigenvecs rescaled with
                   eigenvals, skip first trivial one); each row is
                   data pt  
    k          --- number of clusters
    phi0       --- first left eigenvector of P (prop. to stationary distr)
    Niter      --- number of iterations to repeat the clustering with new IC
    epsilon    --- relative distortion (stopping criteria); default: 0.001
     
    Output:
    idx(n,1)   --- labeling of data points; labels between 1 and k
    C(k,d)     --- geometric centroids
    D          --- distortion (sum of squared distances)
    DX(n,k)    --- squared distance from each point to every centroid

    Calls:       --- distortionMinimization (in each iteration)
     
    Ann B. Lee, May 2008. Last changed, JWR: 3/23/2009'''
 
    N,d = X.shape
    aD = nu.inf # maximum distortion


    #--------------------------------------
    # K-MEANS (repeat Niter times)
    #--------------------------------------
    for iter in range(Niter):
        tmp_ind = nu.ceil(nu.random.rand(k,1)*N)  # k random points in X
        c_0 = X[tmp_ind,:]   # k-by-d matrix of initial centroids
        idx,c,cindex,D,DX = distortionMinimization(X,phi0,k,c_0,0,epsilon)
        if D < aD: #keep best result
            aD,aDX=D,DX
            a_idx = idx
            ac = c

    D,DX,idx,C = aD,aDX,a_idx, ac        

    return idx, C, D, DX


def distortionMinimization(X, phi0, k, c_0=None, DspFlag=0, epsilon=10**-3):
    '''distortionMinimization(X,phi0,k,c_0,DspFlag,epsilon);
    
    Input:
    X(n,d)        --- diffusion coordinates (right eigenvecs
                    rescaled with eigenvals); each row is a data pt
    phi0(n,1)     --- stationary distribution (first left
                      eigenvector of Markov matrix) 
    k             --- number of clusters
    c_0(k,d)      --- initial centers
    DspFlag       --- display flag
    epsilon       --- relative distortion (stopping criteria)
    
    Output:
    S(n,1)        --- labeling; n-dim vector with numbers between 1 and k
    c(k,d)        --- geometric centroids
    cindex(k,1)   --- diffusion centers -- subset of original data; 
                      k-dim vector (row indices of X)
    D             --- distortion
    DX(n,k)       --- squared distance from each point to every centroid
    
    Original version by Stephane Lafon, Yale University, April 2005.
    Last modified, Ann B. Lee, 4/11/05. ABL at CMU: 5/7/08 '''

    #global cPoints  # Warning: global variable

    n,d = X.shape
    if c_0 is None:
        tmp_ind = nu.ceil(nu.random.rand(k,1) * n)  # random subset of X
        c_0 = X[tmp_ind,:]   # k-by-d matrix
        #S_0 = nu.ceil(nu.random.rand(n,1) * k);
'''
    col='rgkbmcy'
    %S = S_0
    %c = nu.zeros((k,d))
    c = c_0
    oldD = nu.inf

    MaxIter=1000;

    for i in range(MaxIter):  # KMEANS LOOP
        
        #-----------------------------------
        # K-MEANS
        #-----------------------------------
        # Update distances to centroids and labels of data points:
        #DX = []    
        for j in range(k):
            dX = X - nu.ones((n,1)) * c[j,:]  # n-by-d
            DX = [DX, nu.dot(dX.T,dX.T).T]  T enter column wise; n-by-k matrix of distances
    
        Dtmp,j = nu.argmin(DX.T)  # min(DX,2)  -> 1-by-n
        Dtmp = DX.T[j]
        S = j # new labels
        #D=Dtmp*phi0; % distortion (a number)
        # Check for empty clusters:
        for j=1:k,
        ind=find(S==j);
        if(isempty(ind)) % if cluster j is empty,
            [mx,m]=max(Dtmp); % find data point that is furthest from its centroid
            S(m)=j; % assign this point the label j
            Dtmp(m)=0;
            %c(j,:) = X(ind(m),:);  % make this point centroid j (redundant)
        end
    end    
    
    % Update centroids:
    for j=1:k,
        ind=find(S==j);
        c(j,:)=phi0(ind,1)'*X(ind,:)/sum(phi0(ind,1)); % find centroid of cluster j
        %plot(c(j,1),c(j,2),'k*');
    end
    
    % Distortion
    D=Dtmp*phi0;
    
    % Plot results:
    if DspFlag
        figure(1), clf, scatter(cPoints(:,1),cPoints(:,2),15,S); axis image, hold on;
        for j=1:k,
            dX=X-ones(n,1)*c(j,:);  % n-by-d
            DX=dot(dX',dX')';  % n-dim vector
            [dummy,tmpind]=min(DX,[],1);
            cindex(j)=tmpind;
            plot(cPoints(cindex(j),1),cPoints(cindex(j),2),'rx'); % diffusion centers
        end
        hold off;
        pause,
    end
     
    % Stopping criteria:
    if((oldD-D)/D < epsilon)
        break;
    end
    oldD=D;
end

%------------------------------------------------------------
% centroids => "diffusion centers" (subset of original data)
%------------------------------------------------------------
cindex=zeros(k,1);
for j=1:k,
    %ind=find(S==j);
    %dX=X(ind,:)-ones(length(ind),1)*c(j,:);
    dX=X-ones(n,1)*c(j,:);  % n-by-d
    DX=dot(dX',dX')';  % n-dim vector
    [dummy,tmpind]=min(DX,[],1);
    cindex(j)=tmpind;
    %DX=dot(dX',dX')';
    %[tmp,i]=min(DX);
    %cindex(j)=ind(i);
end
'''
