import tensorflow as tf
from tensorflow.keras import layers, regularizers
import lmbspecialops as sops
from .tfutils import *


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

def reshape_into(inputs, input_to_copy):
    out = lambda x:x
    off=0
    if len(inputs.shape)==3:
        off = -1
        inputs = tf.expand_dims(inputs,axis=0)
        out = lambda x:tf.squeeze(x,axis=0)
    return out(tf.image.resize_bilinear(inputs, [input_to_copy.get_shape()[1+off].value,
                                             input_to_copy.get_shape()[2+off].value], align_corners=True))

def bi_downsampling(inputs, scale):
    out = lambda x:x
    if len(inputs.shape)==3:
        inputs = tf.expand_dims(inputs,axis=0)
        out = lambda x:tf.squeeze(x,axis=0)
    return out(tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1] // scale, 
                                    tf.shape(inputs)[2] // scale],
                                    align_corners=True))    
def nn_downsampling(inputs, scale):
    out = lambda x:x
    if len(inputs.shape)==3:
        inputs = tf.expand_dims(inputs,axis=0)
        out = lambda x:tf.squeeze(x,axis=0)
    return out(tf.image.resize_nearest_neighbor(inputs, size=[tf.shape(inputs)[1] // scale, 
                                    tf.shape(inputs)[2] // scale],
                                    align_corners=True))    

def replace_nonfinite(has_nans):
    return tf.where(tf.logical_not(tf.is_finite(has_nans)), 
                        tf.zeros_like(has_nans), 
                        has_nans)

def l1_loss(x, epsilon):
    """L1 loss

    Returns a scalar tensor with the loss
    """
    with tf.name_scope("l1_loss"):
        return tf.reduce_sum(tf.sqrt(x**2 + epsilon))

def l2_loss(inp, gt, epsilon,mask = None):
    """L1 loss

    Returns a scalar tensor with the loss
    """
    with tf.name_scope('l2_loss'):
        gt_ = tf.stop_gradient(gt)
        diff = sops.replace_nonfinite(inp-gt_)
        if mask is not None:
            while len(mask.shape)<len(diff.shape):
                mask = tf.expand_dims(mask,-1)
            diff = mask*diff
        return tf.reduce_mean(tf.sqrt(tf.reduce_sum(diff**2, axis=[1,2,3])+epsilon))
   
def pointwise_l2_loss(inp, gt, epsilon=0.00001,mask = None):
    """Computes the pointwise unsquared l2 loss. One channel is equal to l1
    The input tensors must use the format NCHW. 
    This loss ignores nan values. 
    The loss is normalized by the number of pixels.
    
    inp: Tensor
        This is the prediction.
        
    gt: Tensor
        The ground truth with the same shape as 'inp'
        
    epsilon: float
        The epsilon value to avoid division by zero in the gradient computation
    """
    
    with tf.name_scope('pointwise_l2_loss'):
        gt_ = tf.stop_gradient(gt)
        diff = sops.replace_nonfinite(inp-gt_)
        if mask is not None:
            while len(mask.shape)<len(diff.shape):
                mask = tf.expand_dims(mask,-1)
            diff = mask*diff
    
        return tf.reduce_mean(tf.sqrt(tf.reduce_sum(diff**2, axis=3)+epsilon))


def compute_depth_confidence_map(predicted_depth, gt_depth, scale=1):
    """Computes the ground truth confidence map as c_gt = exp(-s|f_pr-f_gt|) 
    
    predict_depth: Tensor
        The predicted flow
        
    gt_depth: Tensor
        The ground truth flow

    scale: float
        Scale factor for the absolute differences
    """
    with tf.name_scope('compute_confidence_map'):
        return tf.exp(-scale*tf.abs(predicted_depth - gt_depth))

def scale_invariant_gradient( inp, deltas=[1,2,4,8,16],
                                weights=[1,1,1,1,1],epsilon=0.001):
    """Computes the scale invariant gradient images
    
    inp: Tensor
        
    deltas: list of int
      The pixel delta for the difference. 
      This vector must be the same length as weight.

    weights: list of float
      The weight factor for each difference.
      This vector must be the same length as delta.

    epsilon: float
      epsilon value for avoiding division by zero
        
    """
    inbatch = len(inp.shape.as_list())==4
    if not inbatch:
        inp = tf.expand_dims(inp,0)
    inp = convert_NHWC_to_NCHW(inp)
    assert len(deltas)==len(weights)
    sig_images = []
    for delta, weight in zip(deltas,weights):
        sig_images.append(sops.scale_invariant_gradient(inp, deltas=[delta], weights=[weight], epsilon=epsilon))
    output = tf.concat(sig_images,axis=1)
    output = convert_NCHW_to_NHWC(output)
    if not inbatch:
        output = tf.squeeze(output,0)
    return output
_k=1000.
def compute_losses(data,predictions,reduction_factor = 0.6,
    depth_w=150., sig_w=500, conf_w = 50., 
    conf_scale = 10., normal_w = 20, iters_to_sig = 2*_k,
    max_iter=300,global_stepf=0,training=True,
    epsilon=0.0001,**kwargs):
    
    total_loss = 0
    weight = 1.
    sig_scale=1.
    if training:
        sig_scale = ease_out_quad(global_stepf-iters_to_sig, 0, 1, float(max_iter//3))
    loss_dict={}
    for i in range(len(predictions)):
        lkr='losses/'+str(i)+'/' # loss key root
        p = predictions[i]
        if p is not None and p!={}:
            _depth_loss = 0
            _sig_loss = 0
            _scaled_sig_loss = 0
            _conf_loss = 0
            _normal_loss = 0

            _depth_gt = data['gt']['depth'+str(i)]
            _sig_gt = data['gt']['sig_depth'+str(i)]
            _norm_gt = data['gt']['normal'+str(i)]
            #DEPTH:
            if 'D' in p:
                _depth_loss = pointwise_l2_loss(p['D'],_depth_gt,epsilon=epsilon)
                _sig_pred = scale_invariant_gradient(p['D'])
                _sig_loss = pointwise_l2_loss(_sig_pred,_sig_gt,epsilon=epsilon)
                _scaled_sig_loss = _sig_loss*sig_scale
                loss_dict[lkr+'depth']=_depth_loss
                loss_dict[lkr+'sig']=_sig_loss
                loss_dict[lkr+'scaled_sig']=_scaled_sig_loss
            #NORMALS:
            if 'N' in p:
                _normal_loss = pointwise_l2_loss(p['N'],_norm_gt,epsilon=epsilon)
                loss_dict[lkr+'normal']=_normal_loss
            #CONFIDENCE:
            if 'C' in p and 'D' in p:
                _conf_gt=compute_depth_confidence_map(p['D'], _depth_gt, scale=conf_scale)
                _conf_loss = pointwise_l2_loss(p['C'],_conf_gt,epsilon=epsilon)
                loss_dict[lkr+'conf']=_conf_loss

            _total_loss_res = depth_w  * _depth_loss 
            _total_loss_res = _total_loss_res + sig_w    * _scaled_sig_loss 
            _total_loss_res = _total_loss_res + normal_w * _normal_loss 
            _total_loss_res = _total_loss_res + conf_w   * _conf_loss
            loss_dict[lkr+'total']= _total_loss_res

            total_loss = total_loss + weight * _total_loss_res
            weight = weight*reduction_factor 
    loss_dict['losses/TOTAL_LOSS']=total_loss
    return total_loss,loss_dict


def PrepareGt(**kargs):
    return lambda x: prepare_gt(x,**kargs)
def prepare_gt(data,
    from_inverse_depth = False, 
    to_inverse_depth = False,
    disparity_map = False,
    data_format='channels_last', # TODO: Support channels first is not implemented yet
    focal_norm = False,
    computate_sig=True,
    img_keys = ['image'], 
    depth_keys = ['depth'], normal_keys = [],
    compute_normals = True, # TODO: Support online normal estimation
    focal_factor = 100.,
    downsampling_depth=5,
    sig_params = None,**kargs):
    
    gt={}
    if sig_params is None:
        sig_params = {'deltas':[1,2,4,8,16], 'weights':[1,1,1,1,1], 'epsilon': 0.001}
    
    K = data['intrinsics']
    w = data['depth'].shape[-2].value
    h = data['depth'].shape[-3].value

    focal_w = K[0,0]
    focal_h = K[0,1]
    ppw = K[0,2]
    pph = K[0,3]
    print([h,w])
    print([focal_w,focal_h, ppw,pph])
    print([focal_w/w,focal_h/h, ppw/w,pph/h])
    w,h=float(w),float(h)
    intrinsics = tf.math.divide(K,tf.convert_to_tensor([[w,h,w,h]]))
    # Prepare depth images at different scales:
    norm_mul = 1.


    if focal_norm:
        focal_orig = ((K[0,0]+K[0,1])/2.)
        norm_mul = focal_factor/focal_orig
        if to_inverse_depth:
            norm_mul = tf.reciprocal(norm_mul)

    for dk in depth_keys:
        depth = data[dk]*norm_mul
        if (from_inverse_depth ^ to_inverse_depth): # XOR (true if different)
            depth=tf.reciprocal(depth)
        for i in range(downsampling_depth):
            gt[dk+str(i)]=depth
            if computate_sig:
                gt['sig_'+dk+str(i)]=scale_invariant_gradient(depth,**sig_params)
            if compute_normals:
                infn = lambda x: convert_NHWC_to_NCHW(tf.expand_dims(x,0))
                outfn = lambda x: tf.squeeze(convert_NCHW_to_NHWC(x),0)
                gt['norm_'+dk+str(i)]=outfn(
                    sops.depth_to_normals(infn(depth),
                    intrinsics,inverse_depth=to_inverse_depth))
            depth = nn_downsampling(depth,2)
    # Prepare normal images at different scales:
    for nk in normal_keys:
        normals = data[nk]
        for i in range(downsampling_depth):
            gt[nk+str(i)]=normals
            normals = nn_downsampling(normals,2)

    data['gt']=gt
    return data
