import tensorflow as tf 

# Some helper functions
# ed = expand last dim
ed = lambda x: tf.expand_dims(x,-1)
ed2 = lambda x: ed(ed(x))
ed3 = lambda x: ed(ed2(x))

def convert_NCHW_to_NHWC(inp):
    """Convert the tensor from caffe format NCHW into tensorflow format NHWC
        
        inp: tensor 
    """
    return tf.transpose(inp,[0,2,3,1])

def convert_NHWC_to_NCHW(inp):
    """Convert the tensor from tensorflow format NHWC into caffe format NCHW 
        
        inp: tensor 
    """
    return tf.transpose(inp,[0,3,1,2])


