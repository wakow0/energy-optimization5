import tensorflow as tf

# Check available GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("TensorFlow is using the GPU.")
else:
    print("TensorFlow is using the CPU.")

