import numpy as np

def make_datastream(dataset, batchsize=100, buffer_size=1000):
    """
    Make the dataset iterator you probably want.
    """
    import tensorflow as tf
    repeat_dataset = dataset.repeat()
    shuffled_dataset = repeat_dataset.shuffle(buffer_size=buffer_size)
    batched_dataset = shuffled_dataset.batch(batchsize)
    iterator = batched_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    return tf.stack(next_element)



class Scaler():
    """
    Helper class to manage scaling a neural network to the nominal range.
    TODO: Tensorflowify this to apply it to a model automagically.
    """
    def __init__(self, data):
        self.scale = []
        ndata = np.empty(data.shape)
        for i in xrange(data.shape[1]):
            x = data[:,i]
            self.scale.append( [np.mean(x), np.amin(x), np.amax(x)] )
            ndata[:,i] = (x - self.scale[-1][0]) /\
              (self.scale[-1][2] - self.scale[-1][1])
        self.scaled_data = ndata
        
    def apply(self, x, i):
        y = np.empty(x.shape)
        for j in xrange(x.shape[-1]):
            y[:,j] = (x[:,j] - self.scale[i+j][0] ) /\
              ( self.scale[i+j][2]-self.scale[i+j][1] )
        return y
    
    def invert(self, x, i):
        y = np.empty(x.shape)
        for j in xrange(x.shape[-1]):
            y[:,j] = x[:,j] * ( self.scale[i+j][2]-self.scale[i+j][1] ) \
              + self.scale[i+j][0]
        return y

def Divy(data, xslice, yslice, rtest, rvalid=0):
    """
    Divy data up into groups for training. Only works on numpy.
    """
    # Prep the training data
    n = data.shape[0]
    idxs = np.array(range(n)).astype(np.int32)
    np.random.shuffle(idxs)
    
    nTestIdxs = int(n * rtest)
    nValidIdxs = int(n * rvalid)
    validIdxs = idxs[0:nValidIdxs]
    testIdxs = idxs[nValidIdxs:nValidIdxs + nTestIdxs]
    nTrainIdxs = n - nValidIdxs - nTestIdxs
    trainIdxs = idxs[nValidIdxs + nTestIdxs:n]
    print('Training data points: %d' % nTrainIdxs)
    print('Testing data points: %d' % nTestIdxs)
    print('Validation data points: %d' % nValidIdxs)
    train_x = data[trainIdxs,xslice]
    test_x  = data[testIdxs, xslice]
    valid_x = data[validIdxs,xslice]
    train_y = data[trainIdxs,yslice]
    test_y  = data[testIdxs, yslice]
    valid_y = data[validIdxs,yslice]
    if len(train_x.shape)<2:
        train_x = train_x.reshape(train_x.size,1)
        test_x = test_x.reshape(test_x.size,1)
        valid_x = valid_x.reshape(valid_x.size,1)
    if len(train_y.shape)<2:
        train_y = train_y.reshape(train_y.size,1)
        test_y = test_y.reshape(test_y.size,1)
        valid_y = valid_y.reshape(valid_y.size,1)
        
    return train_x,train_y, test_x,test_y, valid_x,valid_y
