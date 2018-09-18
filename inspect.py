import tensorflow as tf


def travel_op(op,indents=0):
    "Print a tree from a tensorflow op"
    # TODO: Detect op or tensor
    for _ in xrange(indents): print " ",
    print op.name, " : ", op.type
    for i in op.inputs:
        travel_op(i.op,indents+2)
        
def write_trimmed_meta_graph(graph,sess,
                             outputs,# input_node,
                             ofile):
    """
    Take a graph description, trim it down, and write it
    """
    output_graph_def \
        = tf.graph_util.convert_variables_to_constants(
            sess,graph.as_graph_def(),outputs)
    sub_output = tf.graph_util.extract_sub_graph(
        output_graph_def, outputs)
    newgraph=tf.Graph()
    with newgraph.as_default():
        tf.import_graph_def(sub_output,
                            name='',
                            return_elements=outputs)
        meta_graph_def = tf.train.export_meta_graph(filename=ofile,
                                                    graph_def=newgraph.as_graph_def())

        
def write_trimmed_pb_graph(graph,sess,
                           outputs,# input_node,
                           ofile):
    """
    Take a graph description, trim it down, and write it
    """
    output_graph_def \
        = tf.graph_util.convert_variables_to_constants(
            sess,graph.as_graph_def(),outputs)
    sub_output = tf.graph_util.extract_sub_graph(
        output_graph_def, outputs)

    from tensorflow.python.platform import gfile
    with gfile.GFile(ofile, "wb") as f:
        f.write(sub_output.SerializeToString())
