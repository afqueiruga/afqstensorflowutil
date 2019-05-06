import numpy as np
import itertools
import tensorflow as tf
from functools import reduce

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

def shape_to_size(Y):
    """Calculate the total size of the tensor my multiplying the shapes"""
    return int(reduce(lambda a,b:a*b,Y.shape))

def flatpack(Y):
    """Pack a list of tensors"""
    return tf.concat([tf.reshape(y, (-1,)) for y in Y],axis=-1)


def vector_gradient_dep(y, x):
    """
    Take a gradient with more reasonable behavior. 
    tf.gradients is problematic in its handling of higher rank targets.
    """
    yl = tf.unstack(y,axis=1)
    gl = [ tf.gradients(_,x)[0] for _ in yl ]
    return tf.transpose(tf.stack(gl,axis=-1),perm=[0,2,1])

def vector_gradient(y, x):
    """
    Take a gradient with more reasonable behavior. 
    tf.gradients is problematic in its handling of higher rank targets.
    """
    yl = tf.unstack(y,axis=0)
    gs = [ tf.gradients(_,x)[0] for _ in yl ]
    return tf.stack([tf.reshape(g,(-1,)) for g in gs])


def NewtonsMethod_dep(P, x, alpha=1.0):
    """
    Deprecated:Gives you an operator that performs standard Newton's method
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


def NewtonsMethod2_single_tensor(P, x, alpha=1.0):
    """
    Gives you an operator that performs standard Newton's method.
    Only operates on one tensor.
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
    
def NewtonsMethod(P, x, alpha=1.0):
    """
    Gives you an operator that performs standard Newton's method
    """
    try: # is a single entry
        N = shape_to_size(x)
        x = [x]
    except AttributeError: # is a list
        N = sum([shape_to_size(_x) for _x in x])
    packed_ranges = []
    ptr = 0
    for y in x:
        packed_ranges.append( (ptr, ptr+shape_to_size(y)) )
        ptr += shape_to_size(y)
    print(packed_ranges)
    assert ptr==N

    Grad = tf.gradients(P,x)
    Grad_flat = flatpack(Grad)
    hs = [ flatpack(tf.gradients(_,x)) for _ in tf.unstack(Grad_flat) ]
    Hess = tf.stack([tf.reshape(h,(-1,)) for h in hs])
    # TensorFlow f's up the shapes a lot, so I'm constantly reshaping
    delta = tf.matrix_solve(Hess, tf.expand_dims(Grad_flat,1))
#     delta_reshaped = tf.reshape(delta, x.shape)
    print(delta.shape)
    delta_split = [ tf.slice(delta, (beg,0),(end-beg,1)) for beg,end in packed_ranges ]
    ops = [
        y.assign_add(-alpha*tf.reshape(delta_y,y.shape))
        for delta_y,y in zip(delta_split,x)
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
    for i in range(a.shape[-1]):
        for j in range(i if triangle else 0,b.shape[-1]):
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
    for i in range(3,o+1):
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
    return int(sum([ choose( dim+i-1, dim-1 ) for i in range(1,o+1) ]))
