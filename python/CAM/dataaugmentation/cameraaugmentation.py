import numpy as np
import tensorflow as tf


def deg2rad(deg):
    pi_on_180 = 0.017453292519943295
    return deg * pi_on_180


def rad2deg(rad):
    pi_on_180 = 0.017453292519943295
    return rad / pi_on_180


def stack_CAMDAUG(data,list_augs,**kargs):
    f = data
    for augop in list_augs:
        f = augop(**kargs)(f) 
    return f

class CameraDataAugment():
    def __init__(self,function):
        self.aug = function
        pass
    def __call__(self,x):
        return self.aug(x)



class SaveOriginalSetup(CameraDataAugment):
    def __init__(self,**kwargs):
        super(SaveOriginalSetup,self).__init__(self._saveOriginal)
    @staticmethod
    def _saveOriginal(data):
        original_data={}
        for k,v in data.items():
            original_data[k]=v
        data['original']=original_data
        return data



class Normalization(CameraDataAugment):
    def __init__(self,key_norms={
                'image':[('set_shape','w','h',3),
                         ('cast',tf.float32),
                         ('range',0.,255.,0.,1.),
                         ('custom',lambda x:x)],
                'normal':[('set_shape','w','h',3),
                         ('set_nan',(128,128,128)),
                         ('cast',tf.float32),
                         ('range',0.,255.,-127.5,127.5),
                         ('custom',lambda x:x)],
                'depth':[('set_shape','w','h',1),
                         ('cast',tf.float32),
                         ('custom',lambda x:x)],
                'intrinsics':[('custom',lambda x: tf.expand_dims(x,0) if x.shape == (4) else x)]
                    },
                debug = False,**kargs
                ):
        """
        Normalization operations will be done in order. There are some predefined operations but you can also use your own:
        ('set_shape', w,h,c)
            set shape to w,h,c
                w,h,c can be integers or strings, if strings it will be considered as key in the input dictionary.
        ('cast',type)
            cast to datatype type
        ('range',om,oM,tm,tM)
            normalize values from [om,oM] (original min and MAX) to the target range [tm,tM] 
        ('set_nan',if_condition)
            set values to nan if they have an expecific value
        ('custom',fun)
            custom function to be applied to the tensor
        """
        self.debug = debug
        self.key_norms=key_norms
        super(Normalization,self).__init__(self._applyAllNormOps_)
    
    def _applyAllNormOps_(self,data):
        for k,ops in self.key_norms.items():
            data=Normalization._applyNormOps_(data,k,ops)
        return data

    @staticmethod
    def _applyNormOps_(data,key,list_ops):
        x = data[key]
        for op in list_ops:
            op_type = op[0]
            if op_type == 'set_shape':
                print(op)
                w,h,c = tuple([op[i+1] if isinstance(op[i+1],int) else data.get(op[i+1],None) for i in range(len(op)-1)])
                #print(h,w,c)
                if isinstance(h,int) and isinstance(w,int) and isinstance(c,int):
                    x.set_shape((h,w,c))
                else:
                    print('[WARNING] Shape could not be set. Values are not integers:',w,h,c,x)
                #print('X',x)
            elif op_type == 'cast':
                x = tf.cast(x,op[1])
            elif op_type == 'range':
                om,oM,tm,tM = tuple(list(op)[1:])
                x = (x-om)/(oM-om)
                x = x*(tM-tm)+tm
            elif op_type == 'set_nan':
                mask = tf.ones_like(x,tf.bool)
                values = op[1]
                for i,v in enumerate(values):
                    cond = tf.equal(x[...,i],v)
                    mask = tf.logical_and(mask,tf.stack([cond for _ in range(len(values))], axis = -1))
                x = tf.where(mask,np.nan+tf.ones_like(x),x)
            elif op_type == 'custom':
                x = op[1](x)
            else:
                print('[WARNING] Normalization Operation {} does not exist'.format(op_type))
        data[key]=x
        return data

class RGBAugment(CameraDataAugment):
    def __init__(self,key_aug={'image':{'hue':25.0/360.}},
                          default_aug={
                          'hue':30.0/360.,
                          'lower_saturation':0.7,'upper_saturation':1.4,
                          'value':0.15, 
                          'bright':0.15,
                          'lower_contrast':0.8,'upper_contrast':1.2,
                          'gamma':0.2},debug=False,**kargs
                         ):
        """
        RGB data augmentation, first operations on HSV will be applied and then in RGB color space.
            First rotation in HUE, saturation multiplication and clipping and

        key_aug : dict
            data keys and its respective data augmentation parameters
        -----params-----
        hue : float
            maximun rotation in hue, e.g set hue to 1 if you want to rotate randomly [-1,1]
        lower_saturarion: float
            minimum value to multiply saturation by
        upper_saturarion: float
            maximum value to multiply saturation by
        value : float
            maximum value change
        lower_contrast : float
            minimum value to multiply RGB by
        upper_contrast : float  
            maximum value to multiply RGB by
        bright : float
            maximum bright addition or substraction
        gamma : float
            maximum gamma value
            
        """
        self.key_aug=key_aug
        self.default_aug = default_aug
        self.debug = debug
        self.kargs = kargs
        
        super(RGBAugment,self).__init__(self._applyAllRGBAug_)
    def _applyAllRGBAug_(self,x):
        x_debug = {}
        for k,k_args in self.key_aug.items():
            if k in x:
                new_kargs={}
                new_kargs.update(new_kargs)
                new_kargs.update(self.default_aug)
                new_kargs.update(k_args)
                x[k],rnd_params=RGBAugment.rgbAugFun(x[k],**new_kargs)
                if self.debug:
                    x_debug[k+'/random_rgb_params']=rnd_params
            else:
                print('[WARNING] RGBAugment._ApplyAllRGBAug_ Key {} Not Found in Data'.format(k))
        x.update(x_debug)
        return x
    
    @staticmethod
    def rgbAugFun(rgb,hue=25.0/360.,
                          lower_saturation=0.7,upper_saturation=1.4,
                          value=0.15, 
                          bright=0.15,
                          lower_contrast=0.8,upper_contrast=1.2,
                          gamma=0.2,**kargs):
        """
        rgb : Tensor
            Image in RGB format, normalized to [0,1], and 'HWC' data format
        """
        r_values ={
        'hue' : tf.random_uniform((),-hue,hue),
        'saturation' : tf.random_uniform((),lower_saturation,upper_saturation),
        'value':tf.random_uniform((),-value,value),
        'contrast':tf.random_uniform((),lower_contrast,upper_contrast),
        'bright':tf.random_uniform((),-bright,bright),
        'gamma':tf.random_uniform((),-gamma,gamma)
        }
        
        hsv = tf.image.rgb_to_hsv(rgb)
        h= hsv[...,0]+r_values['hue']
        h = tf.where(h<0,h+1.,tf.where(h>1,h-1.,h)) # Rotate hue                
        s = tf.clip_by_value(hsv[...,1]*r_values['saturation'],0.,1.)
        v = tf.clip_by_value(hsv[...,2]+r_values['value'],0.,1.)
        hsv = tf.stack([h,s,v],axis=2)
        rgb = tf.image.hsv_to_rgb(hsv)
        rgb = rgb * r_values['contrast']
        rgb = rgb + r_values['bright']
        rgb = tf.clip_by_value(rgb,0,1)
        rgb = tf.clip_by_value(rgb**(1.+r_values['gamma']),0,1)
        return rgb,r_values

                    

class FocalDataAugmentation(CameraDataAugment):
    def __init__(self,target_w=256, target_h=192,img_keys = ['image'], 
                depth_keys = ['depth'], normal_keys = [], 
                min_FOV = 1, max_FOV = np.inf, debug=False,pp_max_shift=30,hmirror = True, **kargs):
        self.target_w = target_w
        self.target_h = target_h
        self.img_keys = img_keys
        self.depth_keys = depth_keys
        self.normal_keys = normal_keys
        self.min_FOV = min_FOV
        self.max_FOV = max_FOV
        self.debug = debug
        self.pp_max_shift = pp_max_shift
        self.hmirror = hmirror
        self.kargs = kargs
        super(FocalDataAugmentation,self).__init__(self._applyFocalAugmentation_)

    def _applyFocalAugmentation_(self,data):
        data = FocalDataAugmentation.principal_point_random_shift(data,self.pp_max_shift,self.img_keys,self.depth_keys,self.normal_keys,self.debug,**self.kargs)
        data = FocalDataAugmentation.focal_aug_by_cropping(data,self.target_w,self.target_h,self.img_keys,self.depth_keys,self.normal_keys,self.min_FOV,self.max_FOV,self.debug,**self.kargs)
        if self.hmirror:
            data = FocalDataAugmentation.h_mirror(data,self.target_w,self.target_h,self.img_keys,self.depth_keys,self.normal_keys,**self.kargs)
        return data

    @staticmethod
    def __define_coord_channels__(n,x_dim,y_dim,data_format='channels_last'):
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
        
        if data_format=='channels_last':
            xx_channel = tf.expand_dims(xx_channel, -1)
            yy_channel = tf.expand_dims(yy_channel, -1)
        else:
            xx_channel = tf.expand_dims(xx_channel, 1)
            yy_channel = tf.expand_dims(yy_channel, 1)
            
        xx_channel = tf.cast(xx_channel, 'float32') 
        yy_channel = tf.cast(yy_channel, 'float32') 
        return xx_channel,yy_channel
    @staticmethod
    def principal_point_random_shift(data, pp_max_shift=30,
                                     img_keys = ['image'], 
                                     depth_keys = ['depth'], normal_keys = [], 
                                     debug=False, **kargs):
        max_shift=pp_max_shift
        r_values={}
        w = data.get('w',None)
        h = data.get('h',None)
        intrinsics = data['intrinsics']
        focal_w = intrinsics[0,0]
        focal_h = intrinsics[0,1]
        ppw = intrinsics[0,2]
        pph = intrinsics[0,3]
        
        
        
        tv_shift = tf.random_uniform((), minval=0,maxval=max_shift/2, dtype=tf.float32)
        bv_shift = tf.random_uniform((), minval=0,maxval=max_shift/2,dtype=tf.float32)

        lh_shift = tf.random_uniform((), minval=0,maxval=max_shift/2, dtype=tf.float32)
        rh_shift = tf.random_uniform((), minval=0,maxval=max_shift/2,dtype=tf.float32)

        r_values['tv_shift']=tf.cast(tf.floor(tv_shift),tf.int32)
        r_values['bv_shift']=tf.cast(tf.floor(bv_shift),tf.int32)
        r_values['lh_shift']=tf.cast(tf.floor(lh_shift),tf.int32)
        r_values['rh_shift']=tf.cast(tf.floor(rh_shift),tf.int32)


        
        height_offset =  r_values['tv_shift']
        width_offset  =  r_values['lh_shift']  
        height_target =  tf.cast(h,tf.int32) - (r_values['tv_shift'] + r_values['bv_shift'])
        width_target  =  tf.cast(w,tf.int32) - (r_values['lh_shift'] + r_values['rh_shift'])

        for k,v in data.items():
            if k in (img_keys+depth_keys+normal_keys):
                v = tf.image.crop_to_bounding_box(v, height_offset, width_offset, 
                                                  height_target, width_target)
            data[k]=v

        ppw = ppw - tf.cast(width_offset,tf.float32)
        pph = pph - tf.cast(height_offset,tf.float32)
        new_intrinsics = tf.convert_to_tensor([[focal_w,focal_h, 
                                                ppw,pph]])
        wt = tf.cast(width_target,tf.float32)
        ht = tf.cast(height_target,tf.float32)
        data['intrinsics']=new_intrinsics
        new_intrinsics_norm = tf.convert_to_tensor([[focal_w/wt,
                                                     focal_h/ht, 
                                                     ppw/wt,
                                                     pph/ht]])
        data['intrinsics_norm']=new_intrinsics_norm

        data['h'] = height_target
        data['w'] = width_target
        if debug:
            data['r_values_principalpoint']=r_values
        return data
    @staticmethod
    def h_mirror(data, target_w=256, target_h=192,img_keys = ['image'], 
                       depth_keys = ['depth'], normal_keys = [],data_format = 'channels_last',
                       h_mirror_th = 0.5 ,**kargs):
        """
        WARNING: data images must be in the correct dimensions and must be target_w x target_h.
        Otherwise it will not work, and the effects have not been tested.
        """
        r_values={}
        w = target_w
        h = target_h
        intrinsics = data['intrinsics']
        intrinsics_norm = data['intrinsics_norm']
        focal_w = intrinsics[0,0]
        focal_h = intrinsics[0,1]
        ppw = intrinsics[0,2]
        pph = intrinsics[0,3]

        xx_channel,yy_channel = FocalDataAugmentation.__define_coord_channels__(1,x_dim=w,y_dim=h,data_format=data_format)
        mirror = tf.random_uniform((), minval=0,maxval=1, dtype=tf.float32)
        xx_channel_mirrored = tf.expand_dims(tf.image.flip_left_right(tf.squeeze(xx_channel,0)),0)

        xx_channel_final = tf.cond(mirror>h_mirror_th,lambda : xx_channel_mirrored, lambda: xx_channel)
        yy_channel_final = yy_channel
        ppw = tf.cond(mirror>h_mirror_th,lambda : w-ppw, lambda: ppw)
        #pph = tf.cond(mirror>h_mirror_th,lambda : h-pph, lambda: pph)

        warp = tf.concat((xx_channel_final,yy_channel_final),axis=-1)


        hmirror_factor = tf.cond(mirror>h_mirror_th,lambda :-1.,lambda :1.)

        for k,v in data.items():
            if k in (img_keys):
                v = tf.contrib.resampler.resampler(tf.expand_dims(v,0), warp)
                v = tf.squeeze(v,0)
            elif k in (depth_keys+normal_keys):
                v = tf.contrib.resampler.resampler(tf.expand_dims(v,0), tf.round(warp))
                v = tf.squeeze(v,0)
                if k in normal_keys:
                    # mirror rotate x-axis of the normals
                    normals_x, normals_y, normals_z = tf.split(v,[1,1,1],2) 
                    normals_x = hmirror_factor*normals_x   
                    v = tf.concat([normals_x,normals_y,normals_z],2)
            data[k]=v
        new_intrinsics = tf.convert_to_tensor([[focal_w,focal_h, 
                                                ppw,pph]])
        new_intrinsics_norm = tf.convert_to_tensor([[focal_w/w,
                                                     focal_h/h, 
                                                     ppw/w,
                                                     pph/h]])
        data['intrinsics']=new_intrinsics
        data['intrinsics_norm']=new_intrinsics_norm

        return data
    @staticmethod
    def resize_data(data,oldw,oldh,rnw,rnh,img_keys = ['image'], 
                    depth_keys = ['depth'], normal_keys = [],debug=False,**kargs):
        """
        Resize images images (image,normals and depth images)
        
        For all the images du image resizing
            - For images set them use area interpolation
            - For normals set them using nearest neighbor
            - For depth, using nearest neighbor
        """
       
        factor_w = tf.cast(rnw,tf.float32)/tf.cast(oldw,tf.float32)
        factor_h = tf.cast(rnh,tf.float32)/tf.cast(oldh,tf.float32)
        
        #### Recalculate INTRINSICS
        intrinsics=data['intrinsics']
        fw=intrinsics[0,0]
        fh=intrinsics[0,1]
        cw=intrinsics[0,2]
        ch=intrinsics[0,3]
        new_intrinsics = tf.convert_to_tensor([[fw*factor_w,fh*factor_h, 
                                                cw*factor_w,ch*factor_h]])
        data['intrinsics']=new_intrinsics
        #### DO Resize
        
        #if debug:
        #    print('[RD] resize_data: Resizing to ',rnw,rnh)
        for k,v in data.items():
            method = None
            if k in img_keys:
                method=tf.image.ResizeMethod.AREA
            elif k in (depth_keys+normal_keys):
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            if method is not None:
                v = tf.image.resize_images(v,[rnh,rnw],method,align_corners=True)
            #if debug:
            #    print(k,v)
            data[k] = v
        #print('[RD] resize_data: Resizing done ',data)
        return data
    @staticmethod
    def focal_aug_by_cropping(data, target_w=256, target_h=192,img_keys = ['image'], 
                    depth_keys = ['depth'], normal_keys = [], 
                    min_FOV = 1, max_FOV = np.inf, debug=False, **kargs):
        
        r_values ={}
        r_values['TARGET']=(target_w,target_h)
        w = data.get('w',None)
        h = data.get('h',None)
        intrinsics = data['intrinsics']
        intrinsics_norm = data['intrinsics_norm']
        curr_focal = 0
        #print(intrinsics)
        focal_w = intrinsics[0,0]
        focal_h = intrinsics[0,1]
        ppw = intrinsics[0,2]
        pph = intrinsics[0,3]
        
        # 1) Calculate angles in radians
        cFOVw = tf.atan(tf.cast(w,tf.float32)/2/focal_w)
        cFOVh = tf.atan(tf.cast(h,tf.float32)/2/focal_h)
        current_FOV = tf.minimum(cFOVw,cFOVh)
        min_alpha = min_FOV/2.0
        min_alpha_rads = tf.minimum(deg2rad(min_alpha),current_FOV)
        r_values['min_alpha_rads']=min_alpha_rads
        
        max_alpha = max_FOV/2.0
        max_alpha_rads = deg2rad(max_alpha)
        
        # 2) Calculate max and min possible sides by FOV restrictions
        max_w_by_f = 2*(tf.tan(max_alpha_rads)*focal_w)
        min_w_by_f = 2*(tf.tan(min_alpha_rads)*focal_w)
        max_h_by_f = 2*(tf.tan(max_alpha_rads)*focal_h)
        min_h_by_f = 2*(tf.tan(min_alpha_rads)*focal_h)
        r_values['max_w_by_f']=max_w_by_f
        r_values['min_w_by_f']=min_w_by_f
        r_values['max_h_by_f']=max_h_by_f
        r_values['min_h_by_f']=min_h_by_f
        
        
        # 3) Calculate max and min sides possible by FOV and image restrictions
        max_possible_w = tf.minimum(tf.cast(w,tf.float32),max_w_by_f)
        max_possible_h = tf.minimum(tf.cast(h,tf.float32),max_h_by_f)
        min_possible_w = tf.maximum(tf.cast(target_w,tf.float32),min_w_by_f)
        min_possible_h = tf.maximum(tf.cast(target_h,tf.float32),min_h_by_f)
        r_values['max_possible_w']=max_possible_w
        r_values['max_possible_h']=max_possible_h
        r_values['min_possible_w']=min_possible_w
        r_values['min_possible_h']=min_possible_h
        
        # 4) Calculate max and min sizes by target aspect ratio and previous calcs
        max_w_by_max_h = max_possible_h/target_h*target_w
        max_h_by_max_w = max_possible_w/target_w*target_h
        min_w_by_min_h = tf.minimum(min_possible_h/target_h*target_w,tf.cast(w, 
                                  tf.float32))
        min_h_by_min_w = tf.minimum(min_possible_w/target_w*target_h,tf.cast(h, 
                                  tf.float32))
        r_values['max_w_by_max_h']=max_w_by_max_h
        r_values['max_h_by_max_w']=max_h_by_max_w
        r_values['min_w_by_min_h']=min_w_by_min_h
        r_values['min_h_by_min_w']=min_h_by_min_w
        
        # 5) Merge and estimate final min and max
        max_possible_w = tf.minimum(max_possible_w,max_w_by_max_h)
        max_possible_h = tf.minimum(max_possible_h,max_h_by_max_w)
        min_possible_w = tf.minimum(tf.maximum(min_possible_w,min_w_by_min_h), 
                                    max_possible_w)
        min_possible_h = tf.minimum(tf.maximum(min_possible_h,min_h_by_min_w), 
                                    max_possible_h)
        r_values['max_w']=max_possible_w
        r_values['max_h']=max_possible_h
        r_values['min_w']=min_possible_w
        r_values['min_h']=min_possible_h
        
        
        
        # 5) Calculate randomly the crop size
        r_values['orig_intrinsics'] = intrinsics
        r_values['orig_intrinsics_norm'] = intrinsics_norm
        r_values['orig_w'] = w
        r_values['orig_h'] = h
        
        r_values['crop_w'] = tf.random_uniform((), minval=min_possible_w, 
                                               maxval=max_possible_w, 
                                               dtype=tf.float32)
        r_values['crop_h'] = tf.round(r_values['crop_w']/target_w * target_h)
        
        # 6) CROP and Recalculate INTRINSICS
        height_offset = tf.cast(tf.floor((tf.cast(h,tf.float32)-r_values['crop_h'])/2),
                                tf.int32)
        width_offset = tf.cast(tf.floor((tf.cast(w,tf.float32)-r_values['crop_w'])/2),
                               tf.int32)           
        height_target = tf.cast(tf.round(r_values['crop_h']), tf.int32)
        width_target = tf.cast(tf.round(r_values['crop_w']), tf.int32)
        r_values['height_offset'] = height_offset
        r_values['width_offset'] = width_offset
        r_values['height_target'] = height_target
        r_values['width_target'] = width_target
        for k,v in data.items():
            if k in (img_keys+depth_keys+normal_keys):
                v = tf.image.crop_to_bounding_box(v, height_offset, width_offset, 
                                                  height_target, width_target)
            data[k]=v
        
        ppw = ppw - tf.cast(width_offset,tf.float32)
        pph = pph - tf.cast(height_offset,tf.float32)
        new_intrinsics = tf.convert_to_tensor([[focal_w,focal_h, 
                                                ppw,pph]])
        data['intrinsics']=new_intrinsics
        wt = tf.cast(width_target,tf.float32)
        ht = tf.cast(height_target,tf.float32)
        new_intrinsics_norm = tf.convert_to_tensor([[focal_w/wt,
                                                     focal_h/ht, 
                                                     ppw/wt,
                                                     pph/ht]])
        data['intrinsics_norm']=new_intrinsics_norm
        data=FocalDataAugmentation.resize_data(data,width_target,height_target,target_w,target_h,img_keys, 
                         depth_keys, normal_keys,debug=debug,**kargs)
        if debug:
            data['r_values_crop']=r_values
        return data
