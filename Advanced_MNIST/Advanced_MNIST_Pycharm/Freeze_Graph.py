import tensorflow as tf
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib

freeze_graph.freeze_graph('advanced_mnist.pbtxt',
                          '',
                          True,
                          'advanced_mnist.ckpt',
                          'y_readout1',
                          'save/restore_all',
                          'save/Const:0',
                          'frozen_advanced_mnist.pb',
                          True,
                          '')

input_gragh_def = tf.GraphDef()
with tf.gfile.Open('frozen_advanced_mnist.pb', 'rb') as f:
    data = f.read()
    input_gragh_def.ParseFromString(data)

output_graph_def = optimize_for_inference_lib.optimize_for_inference(
    input_gragh_def,
    ['x_input', 'keep_prob'],
    ['y_readout1'],
    tf.float32.as_datatype_enum)

f = tf.gfile.FastGFile('optimized_advanced_mnist.pb', 'w')
f.write(output_graph_def.SerializeToString())
