import numpy as np
import itertools
import tensorflow as tf

def CatVariable(shapes, stddev=0.0):
    """Makes one stacked variable and gives you subslices. Useful for 
    building the gradients and hessians needed for second-order methods,
    so that you get the cross-terms between weights."""
    l = 0
    for shp in shapes:
        il = 1
        for s in shp: il *= s
        l += il
    # V = tf.Variable(tf.zeros(shape=(l,)))
    V = tf.Variable(tf.truncated_normal(shape=(l,), stddev=stddev))
    cuts = []
    l = 0
    for shp in shapes:
        il = 1
        for s in shp: il *= s
        cuts.append(tf.reshape(V[l:(l+il)],shp))
        l += il
    return V, cuts


def vector_gradient(y, x):
    """
    Take a gradient with more reasonable behavior. 
    tf.gradients is problematic in its handling of higher rank targets.
    """
    yl = tf.unstack(y,axis=1)
    gl = [ tf.gradients(_,x)[0] for _ in yl ]
    return tf.transpose(tf.stack(gl,axis=-1),perm=[0,2,1])

def vector_gradient2(y, x):
    """
    Take a gradient with more reasonable behavior. 
    tf.gradients is problematic in its handling of higher rank targets.
    """
    yl = tf.unstack(y,axis=0)
    gs = [ tf.gradients(_,x)[0] for _ in yl ]
    return tf.stack([tf.reshape(g,(-1,)) for g in gs])


def NewtonsMethod(P, x, alpha=1.0):
    """
    Gives you an operator that performs standard Newton's method
    """
    if len(x.shape)!=1:
        raise Exception('')
    N = x.shape[0]
    Grad = tf.gradients(P,x)[0]
    # TensorFlow f's up the shapes a lot, so I'm constantly reshaping
    Hess = tf.reshape(tf.hessians(P,x)[0],shape=[N,N])
    return [
      x.assign_add(-tf.reshape( # Reshaping to have 1 dimension
                   # Never invert!!!!
                   tf.matrix_solve(Hess,
                            tf.expand_dims(Grad,1))# Reshaping to have 2 dimensions
                ,(N,)))
    ]


def NewtonsMethod2(P, x, alpha=1.0):
    """
    Gives you an operator that performs standard Newton's method
    """
    N = x.shape[0]
    
    Grad = tf.gradients(P,x)[0]
    Grad_flat = tf.reshape(Grad, (-1,))
    hs = [ tf.gradients(_,x)[0] for _ in tf.unstack(Grad_flat) ]
    Hess = tf.stack([tf.reshape(h,(-1,)) for h in hs])
    # TensorFlow f's up the shapes a lot, so I'm constantly reshaping
    delta = tf.matrix_solve(Hess, tf.expand_dims(Grad_flat,1))
    delta_reshaped = tf.reshape(delta, x.shape)
    ops = [
        x.assign_add(-alpha*delta_reshaped)
    ]
    return ops
    


def outer(a,b, triangle=False):
    """
    Symbolic outer product:
    stack( a(x)b )
    triangle option toggles whether or not to include symmetric elements (i.e. 
    only return the lower triangle because it's identical to the upper triangle)
    You probably want triangle=True when a==b.
    """
    p = []
    for i in xrange(a.shape[-1]):
        for j in xrange(i if triangle else 0,b.shape[-1]):
            p.append( a[:,i]*b[:,j] )
    return tf.stack(p, axis=-1)

def polyexpand(a,o):
    """
    Build and stack a polynomial basis set of up to exponent o. Includes
    all cross terms. E.g.,
    polyexpand([x y], 2) = [ x y x^2 xy y^2 ]
    """
    if o<=0: raise Exception("I don't know what it means when o<=0")
    if o==1: return a
    p = [a,outer(a,a,True)]
    for i in xrange(3,o+1):
        exponents = itertools.combinations_with_replacement(range(a.shape[-1]),i)
        multinom = []        
        for m in exponents:
            t = a[:,m[0]]
            for e in m[1:]:
                t *= a[:,e]
            multinom.append(t)
        p.append(tf.stack(multinom,axis=-1))
    return tf.concat(p, axis=1)

def Npolyexpand(dim,o):
    """Returns the length of polyexpand to preallocate data."""
    from math import factorial as fac
    choose = lambda n,k : fac(n) / (fac(k)*fac(n-k))
    return sum([ choose( dim+i-1, dim-1 ) for i in range(1,o+1) ])
