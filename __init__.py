from .dataprep import make_datastream
from .Scaler import Scaler
from .operations import CatVariable, vector_gradient, NewtonsMethod, NewtonsMethod_pieces, polyexpand, Npolyexpand
from .operations import vector_gradient_dep
from . import training
from .inspect import write_trimmed_pb_graph, write_trimmed_meta_graph, replicate_subgraph