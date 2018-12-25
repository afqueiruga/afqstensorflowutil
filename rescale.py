import argparse, os
import numpy as np
from . import Scaler

parser = argparse.ArgumentParser(description='Scale a csv.')
parser.add_argument('file', metavar='file', type=str,
        help='file to scale')
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