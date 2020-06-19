from tensorflow.python.framework import graph_util
import tensorflow as tf

graph_def_file = "make_frozen1.pb"
input_arrays = ["content","style"]
output_arrays = ["add_37"]

converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays)
tflite_model = converter.convert()
open("make_2input.tflite", "wb").write(tflite_model)
