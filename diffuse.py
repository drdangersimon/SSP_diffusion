'''Diffusion mapping by Ann B Lee, March 2009. Last modified, JWR: 3/23/2009 ported to python'''



import numpy as nu


def diffuse(D,eps_val,t=None,neigen=None):
    ''' diffuse(D,eps_val,t,neigen)
    (ndarray or matrix, float, float, float
    D          --- pairwise distances, n-by-n matrix
    eps_val    --- parameter in weight matrix exp(-D^2/(4*eps_val))
    neigen     --- number of dimensions of final representation
    (default: 95% drop-off in eigenvalue multiplier)
    t          --- optional time parameter in diffusion map 
    (default: model with multiscale geometry)    

    Output: 
    X          --- non-trivial diffusion coords. (entered column-wise)  
    eigenvals  --- eigenvalues of Markov matrix
    psi        --- right eigenvectors of Markov matrix
    phi        --- left eigenvectors'''

    n = D.size(0)
    K = nu.exp(-D**2/(4.*eps_val)) #or equivalently, K=exp(-(D/sigmaK).^2)
    v = nu.sqrt(nu.sum(K))
    #v = v(:)
    A = K / (v * v.T)   # symmetric graph Laplacian
    threshold = 5E-6 
    #A = sparse(A.*double(A>threshold));  # make matrix sparse to speed up calcs  
    if neigen is None:
        U,S,V = nu.svd(A,51)  # eigendecomposition of symmetric matrix
        psi = U / (U[:,0] * nu.ones((1,51))) # right eigenv of Markov matrix
        phi = U * (U[:,0] * nu.ones((1,51))) # left eigenv of Markov matrix
    else:
        U,S,V = nu.svd(A, neigen+1)  # eigendecomposition of symmetric matrix
        psi = U / (U[:,0] * nu.ones((1,neigen+1))) # right eigenv of Markov matrix
        phi = U * (U[:,1] * nu.ones((1,neigen+1))) # left eigenv of Markov matrix
    
    eigenvals = nu.diag(S)

    # DIFFUSION COORDINATES
    if not t is None: # fixed scale parameter
        lambda_t = eigenvals[1:]**t
        lambda_t = ones(n,1)*lambda_t.T
        if neigen is None: # use neigen corresponding to 95% drop-off in lambda_t
            lam = lambda_t[0,:] / lambda_t[0,0]
            neigen = nu.min(lam[lam < .05]) # default number of eigenvalues
            neigen = nu.min([neigen, 50]) # use neigen=50 if 95% dropoff not attained
            print 'Used default value: %f dimensions'%neigen
  
        X = psi[:,1:neigen] * lambda_t[:,:neigen]  # diffusion coords X
        #= right eigenvectors rescaled with eigenvalues
    else:  # multiscale geomtry
        lambda_multi = eigenvals[1:] / (1-eigenvals[1:])
        lambda_multi = nu.ones((n,1)) * lambda_multi.T
        if neigen is None:  # use neigen corresponding to 95% drop-off in lambda_multi
            lam = lambda_multi[0,:] / lambda_multi[0,0]
            neigen = nu.min(lam[lam<.05]) # default number of eigenvalues
            neigen = nu.min([neigen, 50]) # use neigen=50 if 95% dropoff not attained
            print 'Used default value: %f dimensions'%neigen
        X = psi[:,1:neigen] * lambda_multi[:,:neigen]  # diffusion coords X
        

    return X,eigenvals, psi, phi
