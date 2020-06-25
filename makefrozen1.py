import tensorflow as tf
from tensorflow.python.tools import freeze_graph

freeze_graph.freeze_graph(input_graph='./nonfrozenpbtxt.pbtxt',
                          input_saver="",
                          input_binary=False,
                          input_checkpoint='./final.ckpt',
                          output_node_names='mul_32',
                          restore_op_name="",
                          filename_tensor_name="",
                          output_graph='./make_frozen1.pb',
                          clear_devices=False, initializer_nodes="")
