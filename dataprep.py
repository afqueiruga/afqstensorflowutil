from __future__ import division
import numpy as np
import os

def make_datastream(dataset, batchsize=100, buffer_size=1000):
    """
    Make the dataset iterator you probably want.
    """
    import tensorflow as tf
    repeat_dataset = dataset.repeat()
    shuffled_dataset = repeat_dataset.shuffle(buffer_size=buffer_size)
#     batched_dataset = shuffled_dataset.batch(batchsize)
    iterator = shuffled_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    try:
        stacked = tf.stack(next_element.values())
    except Attribute as e:
        stacked = tf.stack(next_element)
    return stacked

def make_shards(inputfile, outputprefix, entries_per_shards):
    """
    Open a dataset and cut it up into more manageable shards. Each
    shard will be shuffled from the entire dataset.
    """
    # Count the file and get the header
    ndata = 0 # We skip the header
    with open(inputfile,"r") as f:
        header = f.readline()[:-1]
        for l in f:
            ndata+=1
    numfiles = ndata//entries_per_shards + 1

    # Load the dataset into memory
    buffer = np.loadtxt(inputfile,skiprows=1,delimiter=",")
    # Shuffle the indices
    idxs = np.array(range(ndata)).astype(np.int32)
    np.random.shuffle(idxs)
    # split up the indices
    remainder = ndata % entries_per_shards
    location = 0
    os.system('mkdir -p {0}'.format(outputprefix))
    for i in range(numfiles):
        endpoint = location + entries_per_shards + (0 if i < remainder else 1)
        np.savetxt(outputprefix+'/shard_{0}.csv'.format(i), buffer[location:endpoint,:],
                   delimiter=", ", header=header, comments="")
        location = endpoint
    
def make_datastream_from_shards(shardprefix, batchsize):
    """
    Create a shuffled datastream from the shards.
    """
    pass

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
