import argparse, os
import numpy as np
from . import Scaler
from .dataprep import make_shards, divy

"""
A script to normalize and shuffle a csv file.
"""

parser = argparse.ArgumentParser(description='Scale a csv, shuffle, split into train/test/validate sets, and possibly write into shards.')
parser.add_argument('file', metavar='file', type=str,
        help='file to scale')
parser.add_argument('datums_per_shard', metavar='datums_per_shard', type=int,
        help='How many datums should go into a shard')
parser.add_argument('binary', metavar='binary', type=bool,
        help='Output as a csv or a binary format')
args = parser.parse_args()
fname, ext = os.path.splitext(args.file)
path,fname = os.path.split(fname)
path+='/'
with file(path+fname+ext) as f:
    header = f.readline()[:-1]
dat = np.loadtxt(path+fname+ext,skiprows=1,delimiter=",")
    
S = Scaler(dat)

train,test,valid = divy(S.scaled_data, slice(0,4),None, 1.0/6.0,1.0/6.0)



if args.binary:
    np.savez(path+"train_"+fname+"_scaled", train)
    np.savez(path+"test_"+fname+"_scaled", test)
    np.savez(path+"valid_"+fname+"_scaled", valid)
else:
    np.savetxt(path+"train_"+fname+"_scaled"+ext, train,
            delimiter=",",header=header,comments="")
    np.savetxt(path+"test_"+fname+"_scaled"+ext, test,
            delimiter=",",header=header,comments="")
    np.savetxt(path+"valid_"+fname+"_scaled"+ext, valid,
            delimiter=",",header=header,comments="")
    
np.savetxt(fname+"_ranges"+ext,np.array(S.scale).T,
           delimiter=",",header=header,comments="")


if args.datums_per_shard>0:
    make_shards(fname+"_scaled"+ext, fname+"_sharded", args.datums_per_shard)
