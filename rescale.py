import argparse, os
import numpy as np
from . import Scaler
from .dataprep import make_shards

"""
A script to normalize and shuffle a csv file.
"""

parser = argparse.ArgumentParser(description='Scale a csv and shuffle into shards.')
parser.add_argument('file', metavar='file', type=str,
        help='file to scale')
parser.add_argument('datums_per_shard', metavar='datums_per_shard', type=int,
        help='How many datums should go into a shard')
args = parser.parse_args()
fname, ext = os.path.splitext(args.file)

with file(fname+ext) as f:
    header = f.readline()[:-1]
dat = np.loadtxt(fname+ext,skiprows=1,delimiter=",")
    
S = Scaler(dat)

np.savetxt(fname+"_scaled"+ext,S.scaled_data,
           delimiter=",",header=header,comments="")
np.savetxt(fname+"_ranges"+ext,np.array(S.scale).T,
           delimiter=",",header=header,comments="")

if args.datums_per_shard>0:
    make_shards(fname+"_scaled"+ext, fname+"_sharded", args.datums_per_shard)
