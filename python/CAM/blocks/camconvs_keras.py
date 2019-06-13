# 
#
#
#
from tensorflow.keras.layers import Conv2D,Layer
import tensorflow as tf
from .utils import *


class AddCAMCoordsKeras(Layer):
    """ Add Camera Coord Maps to a tensor
    """
    def __init__(self, coord_maps,centered_coord,norm_coord_maps,with_r,bord_dist,scale_centered_coord,fov_maps,
            data_format='channels_last',
            resize_policy=tf.image.ResizeMethod.BILINEAR):
        self.coord_maps = coord_maps
        self.centered_coord = centered_coord
        self.norm_coord_maps = norm_coord_maps
        self.with_r = with_r
        self.bord_dist = bord_dist
        self.scale_centered_coord = scale_centered_coord
        self.fov_maps = fov_maps
        self.data_format=data_format
        self.resize_policy = resize_policy
        super(AddCAMCoordsKeras,self).__init__()

    def additional_channels(self):
        return self.coord_maps*2 + self.centered_coord*2 + self.norm_coord_maps*2 + self.with_r*1 + self.bord_dist*4 + self.fov_maps*2

    def _resize_map_(self,data,w,h):
        if self.data_format == 'channels_first':
            data_cl = convert_NCHW_to_NHWC(data) # data to channels last
            data_cl_r = tf.image.resize_images(data,[h,w],method=self.resize_policy,align_corners = True)
            return convert_NHWC_to_NCHW(data_cl_r)
        else:
            return tf.image.resize_images(data,[h,w],method=self.resize_policy,align_corners = True)
    def __define_coord_channels__(self,n,x_dim,y_dim):
        """
        Returns coord x and y channels from 0 to x_dim-1 and from 0 to y_dim -1
        """
        xx_ones = tf.ones([n, y_dim],dtype=tf.int32)
        xx_ones = tf.expand_dims(xx_ones, -1)
        xx_range = tf.tile(tf.expand_dims(tf.range(x_dim), 0),[n, 1])
        xx_range = tf.expand_dims(xx_range, 1)
        xx_channel = tf.matmul(xx_ones, xx_range)


        yy_ones = tf.ones([n, x_dim],dtype=tf.int32)
        yy_ones = tf.expand_dims(yy_ones, 1)
        yy_range = tf.tile(tf.expand_dims(tf.range(y_dim), 0),[n, 1])
        yy_range = tf.expand_dims(yy_range, -1)
        yy_channel = tf.matmul(yy_range, yy_ones)
        
        if self.data_format=='channels_last':
            xx_channel = tf.expand_dims(xx_channel, -1)
            yy_channel = tf.expand_dims(yy_channel, -1)
        else:
            xx_channel = tf.expand_dims(xx_channel, 1)
            yy_channel = tf.expand_dims(yy_channel, 1)
            
        xx_channel = tf.cast(xx_channel, 'float32') 
        yy_channel = tf.cast(yy_channel, 'float32') 
        return xx_channel,yy_channel

    #def call(self, input_tensor,h,w,cx,cy,fx,fy):
    def call(self, input_tensor,h=0,w=0,cx=0,cy=0,fx=0,fy=0):
        """
        input_tensor: Tensor
            (N,H,W,C) if channels_last or (N,C,H,W) if channels_first
        """
        if self.additional_channels()==0:
            return input_tensor

        batch_size_tensor = tf.shape(input_tensor)[0]
        if self.data_format == 'channels_first':
            x_dim_tensor = tf.shape(input_tensor)[3]
            y_dim_tensor = tf.shape(input_tensor)[2]
            ax_concat = 1
        else:
            x_dim_tensor = tf.shape(input_tensor)[2]
            y_dim_tensor = tf.shape(input_tensor)[1]
            ax_concat = -1
        xx_channel,yy_channel = self.__define_coord_channels__(batch_size_tensor,w,h)

        extra_channels = []
        # 1) Normalized coordinates
        if self.norm_coord_maps:
            norm_xx_channel = (xx_channel/(w-1))*2.0-1.0
            norm_yy_channel = (yy_channel/(h-1))*2.0-1.0
            if self.with_r:
                norm_rr_channel = tf.sqrt(tf.square(norm_xx_channel-0.5)+tf.square(norm_yy_channel-0.5))
                extra_channels = extra_channels + [norm_xx_channel,norm_yy_channel,norm_rr_channel]
            else:
                extra_channels = extra_channels + [norm_xx_channel,norm_yy_channel]

        if self.centered_coord or self.fov_maps:
            # 2) Calculate Centered Coord
            # ed2 is equal to extend_dims twice
            cent_xx_channel = (xx_channel-ed2(cx)+0.5)
            cent_yy_channel = (yy_channel-ed2(cy)+0.5)
            
            # 3) Field of View  coordinates
            if self.fov_maps:
                fov_xx_channel = tf.atan(cent_xx_channel/ed2(fx))
                fov_yy_channel = tf.atan(cent_yy_channel/ed2(fy))
                extra_channels = extra_channels + [fov_xx_channel,fov_yy_channel]
            # 4) Scaled Centered  coordinates
            if self.centered_coord:
                extra_channels = extra_channels + [cent_xx_channel/self.scale_centered_coord,cent_yy_channel/self.scale_centered_coord]
        
        # 5) Coord Maps (Unormalized, uncentered and unscaled)
        if self.coord_maps:
            extra_channels = extra_channels + [xx_channel,yy_channel]
        
        # Concat and resize
        if len(extra_channels)>0:
            extra_channels = tf.concat(extra_channels,axis=ax_concat)  
            extra_channels = self._resize_map_(extra_channels,x_dim_tensor,y_dim_tensor)
            extra_channels = [extra_channels]
        # 6) Distance to border in pixels in feature space.
        if self.bord_dist:
            t_xx_channel,t_yy_channel = self.__define_coord_channels__(batch_size_tensor,x_dim_tensor,y_dim_tensor)
            l_dist = t_xx_channel
            r_dist = tf.cast(x_dim_tensor,tf.float32) - t_xx_channel-1
            t_dist = t_yy_channel
            b_dist = tf.cast(y_dim_tensor,tf.float32) - t_yy_channel-1
            extra_channels = extra_channels + [l_dist,r_dist,t_dist,b_dist]
         

        extra_channels = list(tf.keras.backend.stop_gradient(extra_channels)) # Stop Gradients
        output_tensor = tf.concat(extra_channels+[input_tensor],axis=ax_concat)
        return output_tensor
    

        


class CAMConvKeras(Conv2D, Layer):
    """ Camera-Aware Multiscale Convolution
  

        Big part of this layer inherit from Conv2D, the following comments are in
        part copied from that layer.


        This layer creates a convolution kernel that is convolved
        (actually cross-correlated) with the layer input to produce a tensor of
        outputs. If `use_bias` is True (and a `bias_initializer` is provided),
        a bias vector is created and added to the outputs. Finally, if
        `activation` is not `None`, it is applied to the outputs as well.
        Arguments:
        filters: Integer, the dimensionality of the output space (i.e. the number
          of filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the
          height and width of the 2D convolution window.
          Can be a single integer to specify the same value for
          all spatial dimensions.
        coord_maps : bool 
            the layer will add a Tensor (same size that the image (n,h,w,2)) with the coordinate maps in real values:
                channel 0 :
                                 |0,1,2,3,.....,w|
                     x_coord =   |0,1,2,3,.....,w|
                                 |0,1,2,3,.....,w|
                channel 1:
                                 |0,0,0,0,0,0,0,0|
                     y_coord =   |.,.,.,.,.,.,.,.|
                                 |h,h,h,h,h,h,h,h|
                ** this will be resized with 'resize_policy' to the size of every feature map.
        centered_coord: bool
            if true the coord_maps are 0 centered on the principal point, which meands
                channel 0 :
                                 |0,1,2,3,.....,w|
                     x_coord =   |0,1,2,3,.....,w| - cx
                                 |0,1,2,3,.....,w|
                channel 1:
                                 |0,0,0,0,0,0,0,0|
                     y_coord =   |.,.,.,.,.,.,.,.| - cy
                                 |h,h,h,h,h,h,h,h|
        scale_centered_coord: int
            the centered_coord maps will be divided by this value (in order not to introduce big values into the network., Default 320)
        norm_coord_maps : bool 
            the layer will add a Tensor (same size that the image (n,h,w,2)) with the coordinate maps in real values:
                channel 0 :
                                 |-1,.,.,.,0,.....,1|
                     x_coord =   |-1,.,.,.,0,.....,1|
                                 |-1,.,.,.,0,.....,1|
                channel 1:
                                 |-1,-1,...,-1,-1|
                                 |.,.,.,.,.,.,.,.|
                     y_coord =   |0,0,0,0,0,0,0,0|
                                 |.,.,.,.,.,.,.,.|
                                 |1,1,1,1,1,1,1,1|
                                 
                ** this will be resized with 'resize_policy' to the size of every feature map.
        with_r: bool
            add a channel with the r coordinate presented by Coord Conv 
                r_coord = \sqrt{x_coord^2+y_coord^2}
        bord_dist : bool
            the layer will ad a Tensor with the distance to every border i.e. 4 channels
                Assuming feature map has width = a and height=b
                channel 0 :
                                |a,a-1,....2,1,0|
                     r_dist =   |a,a-1,....2,1,0|
                                |a,a-1,....2,1,0|
                                
                channel 1 :                
                                
                                |0,0,0,0,0,0,0,0|
                                |1,1,1,1,1,1,1,1|
                     t_dist =   |...............|
                                |b-1,b-1,....b-1|
                                |b,b,b,b,b,b,b,b|
                                
                        [...] ; also l_dist and b_dist
                ** this will be calculated for every feature map, as every feature map has a different distance
                
        fov_maps: bool (stands for field.of.view)
            the layer will add a Tensor with the horizontal and  vertical field of view in radians of the camera.
                
                x_fov = tf.atan(centered_coord_maps_x/fx)
                y_fov = tf.atan(centered_coord_maps_y/fy)
            
            ** this will be resized with 'resize_policy' to the size of every feature map. 
        strides: An integer or tuple/list of 2 integers,
          specifying the strides of the convolution along the height and width.
          Can be a single integer to specify the same value for
          all spatial dimensions.
          Specifying any stride value != 1 is incompatible with specifying
          any `dilation_rate` value != 1.
        padding: One of `"valid"` or `"same"` (case-insensitive).
        data_format: A string, one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, height, width, channels)` while `channels_first` corresponds to
          inputs with shape `(batch, channels, height, width)`.
        dilation_rate: An integer or tuple/list of 2 integers, specifying
          the dilation rate to use for dilated convolution.
          Can be a single integer to specify the same value for
          all spatial dimensions.
          Currently, specifying any `dilation_rate` value != 1 is
          incompatible with specifying any stride value != 1.
        activation: Activation function. Set it to None to maintain a
          linear activation.
        use_bias: Boolean, whether the layer uses a bias.
        kernel_initializer: An initializer for the convolution kernel.
        bias_initializer: An initializer for the bias vector. If None, the default
          initializer will be used.
        kernel_regularizer: Optional regularizer for the convolution kernel.
        bias_regularizer: Optional regularizer for the bias vector.
        activity_regularizer: Optional regularizer function for the output.
        kernel_constraint: Optional projection function to be applied to the
            kernel after being updated by an `Optimizer` (e.g. used to implement
            norm constraints or value constraints for layer weights). The function
            must take as input the unprojected variable and must return the
            projected variable (which must have the same shape). Constraints are
            not safe to use when doing asynchronous distributed training.
        bias_constraint: Optional projection function to be applied to the
            bias after being updated by an `Optimizer`.
        trainable: Boolean, if `True` also add variables to the graph collection
          `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
        name: A string, the name of the layer.
    """
    def __init__(self, filters,
               kernel_size,
               coord_maps=False,
               centered_coord = True,
               scale_centered_coord = 320,
               norm_coord_maps = True,
               with_r = False,
               bord_dist = False,
               fov_maps = True,
               strides=(1, 1),
               padding='same',
               data_format='channels_last',
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=tf.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
        self.add_cam_coords = AddCAMCoordsKeras(coord_maps,centered_coord,norm_coord_maps,with_r,bord_dist,scale_centered_coord,fov_maps,data_format=data_format)
        super(CAMConvKeras, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name, **kwargs)
        
    def __call__(self,input_tensor,intrinsics,image_shape,training=None,*args,**kwargs):
        """
            intrinsics:  Tensor 
                intrinsic.shape = (N,4) where N is the batch size and 4 represents the 4 values in order (fx,fy,cx,cy)
            image_shape : TensorShape
                shape of the input image. One of: (N,H,W,C) if channels_last or (N,C,H,W) if channels_first or (N,H,W) or (H,W)
        """
        #print('__call__: input tensor: ',input_tensor)
        assert(image_shape.ndims>=2 and image_shape.ndims<=4)
        if image_shape.ndims == 4:
            _, h, w, _ = image_shape.as_list()
        elif image_shape.ndims == 3:
            _, h, w = image_shape.as_list()
        else:
            h,w = image_shape.as_list()
        fx,fy,cx,cy = tf.split(intrinsics,num_or_size_splits=4,axis=-1)
        new_input_tensor = self.add_cam_coords(input_tensor,h=h,w=w,cx=cx,cy=cy,fx=fx,fy=fy)

        return super(CAMConvKeras,self).__call__(new_input_tensor,*args,**kwargs)

