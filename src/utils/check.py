import tensorflow as tf
print("1. TF Version:", tf.__version__)
print("2. Built with CUDA:", tf.test.is_built_with_cuda())
print("3. GPU Devices:", tf.config.list_physical_devices('GPU'))