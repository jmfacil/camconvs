import functools
import re
from .generator import *
from .datareader import _float_feature,_bytes_feature,_int64_feature,FormatEntry
import json
import os
import numpy as np
import random
import tensorflow as tf
from PIL import Image
class StanfordGenerator(Generator):
    def check_size(self,original_size):
        if self.size is None:
            self.size = original_size
    def load_data(self,path,fn):
        """
        Return the data asociated with fn from a certain path. The type of data will be
        the one specified in self.data, (e.g. ('image','depth'))
        
        path : string
            path to the area folder
        fn : string 
            image identifier without the domain of the file extension.
        """
        if path is None:
            return None
        depthpath = os.path.join(path,'depth')
        normalpath = os.path.join(path,'normal')
        rgbpath = os.path.join(path,'rgb')
        posepath = os.path.join(path,'pose')
        
        output={}
        original_size = None
        reduction_factor = 1
        if 'image' in self.data:
            rgb = Image.open(os.path.join(rgbpath,fn+'_domain_rgb.png')).convert('RGB')
            original_size = rgb.size
            self.check_size(original_size)
            
            if not (self.size == original_size):
                assert self.size[0]/original_size[0] == self.size[1]/original_size[1]
                reduction_factor = self.size[0]/original_size[0]
                rgb=rgb.resize(self.size,resample=Image.LANCZOS)
                
            output['image'] = np.array(rgb)
        
        if 'depth' in self.data:
            depth_png = Image.open(os.path.join(depthpath,fn+'_domain_depth.png'))
            
            original_size = depth_png.size
            self.check_size(original_size)
            if not (self.size == original_size):
                assert self.size[0]/original_size[0] == self.size[1]/original_size[1]
                _reduction_factor = self.size[0]/original_size[0]
                assert (_reduction_factor == reduction_factor) or (reduction_factor == -1)
                reduction_factor = _reduction_factor
                depth_png=depth_png.resize(self.size,resample=Image.NEAREST)
            
            depth_png = np.array(depth_png,dtype=int)
            depth = depth_png.astype(np.float)/512.
            depth[depth>=127.9]=np.NaN
            output['depth'] = depth
            #output['depth_metric'] = 'camera_z'
            
        if 'normal' in self.data:
            try:
                normal = Image.open(os.path.join(normalpath,fn+'_domain_normals.png'))
            except FileNotFoundError:
                normal = Image.open(os.path.join(normalpath,fn+'_domain_normal.png'))
            
            original_size = normal.size
            self.check_size(original_size)
            if not (self.size == original_size):
                assert self.size[0]/original_size[0] == self.size[1]/original_size[1]
                _reduction_factor = self.size[0]/original_size[0]
                assert (_reduction_factor == reduction_factor) or (reduction_factor == -1)
                reduction_factor = _reduction_factor
                normal=normal.resize(self.size,resample=Image.NEAREST)

            normal = np.array(normal)
            output['normal'] = normal
            
        if 'camera_info' in self.data:
            with open(os.path.join(posepath,fn+'_domain_pose.json')) as f:
                data = json.load(f)
                for k in data.keys():
                    if k == 'camera_k_matrix':
                        output['intrinsics'] = np.array(data['camera_k_matrix'])*reduction_factor
                        x = output['intrinsics']
                        output['intrinsics'] =np.array([x[0,0],x[1,1],x[0,2],x[1,2]])
                        k = 'intrinsics'
                        self.ci_format[k]=output[k].shape 
                    elif not(isinstance(data[k],str)):
                        output[k] = np.array(data[k])
                        if not isinstance(data[k],list):
                            output[k] = np.array([data[k]])
                        self.ci_format[k]=output[k].shape 

        if len(set(self.data) & set(('image','depth','normal'))):
            output['w']=self.size[0]
            output['h']=self.size[1]
            output['area']=int(re.search('area_([0-9])',path).group(1))


        return output
    def gen_list_files(self):
        depthpath = os.path.join(self.data_dir,'depth')
        # list of file names
        file_names = os.listdir(depthpath)
        file_names.sort()
        file_names = [name.split('/')[-1].split('_domain')[-2] for name in file_names if '.gitkeep' not in name]
        if self.shuffle:
            random.shuffle(file_names)
        self.files = file_names

    def __init__(self,data_dir,data=('image','depth','normal','camera_info'),
                 shuffle_files = True,size = None,shuffle = True,seed = None):
        random.seed(seed)
        self.data = data
        self.size = size
        self.shuffle = shuffle
        self.data_dir = data_dir
        self.index = -1
        self.depth16bit=False
        if data_dir is not None:
            self.gen_list_files()
            self.number_of_examples = len(self.files)
            self.ci_format = {}
            self.load_data(self.data_dir,self.files[0])


    
    def __call__(self):
        self.index = self.index+1
        if(self.index>=self.number_of_examples):
            raise Exception("All the files have been read already!")
        yield self.load_data(self.data_dir,self.files[self.index])
    def get_number_of_examples(self,):
        return self.number_of_examples
    def format_entry(self,):
        format_db = {}
        if 'image' in self.data:
            format_db['image'] = FormatEntry(
                pre = lambda x: np.array(x), 
                tftype  = tf.uint8,
                tfpre = lambda x: tf.io.encode_jpeg(x),
                featop = lambda x: _bytes_feature(x),
                feat_dtype = tf.string,
                feat_shape =(),
                parse_op = lambda x,**kw: tf.io.decode_image(x,channels=3,dtype=tf.float32) ,
            )
        if 'depth' in self.data:
            def parse_depth(x,w=-1,h=-1,**kw):
                expected_size = w*h
                expected_size = tf.cast(expected_size, dtype=tf.int64)
                depth_shape = tf.concat([[1],[h],[w],[1]],axis=0)
                #print(depth_shape)
                #depth1 = tf.reshape(tf.decode_raw(x, tf.float16 if self.depth16bit else tf.float32), shape=depth_shape)
                depth1 = tf.reshape(x, shape=depth_shape)
                depth1 = tf.cast(depth1, tf.float32)
                depth1 = tf.squeeze(depth1, axis=0)
                return depth1
            format_db['depth'] = FormatEntry(
                pre = lambda x: x.astype(np.float16 if self.depth16bit else np.float32), 
                tftype  = tf.float16 if self.depth16bit else tf.float32,
                tfpre = lambda x: tf.squeeze(x),
                featop = lambda x: _float_feature(x.flatten().tolist()),
                feat_dtype = tf.float16 if self.depth16bit else tf.float32,
                feat_shape = (),
                feat_op = lambda x,*args : tf.FixedLenSequenceFeature(x,*args,allow_missing=True),
                parse_op = lambda x,**kw: parse_depth(x,**kw) ,
                
            )
            
        for k in ['h','w','area']:
            format_db[k] = FormatEntry(
                    pre = lambda x: x, 
                    tftype  = tf.int64,
                    tfpre = lambda x: x,
                    featop = lambda x: _int64_feature(x),
                    feat_dtype = tf.int64,
                    feat_shape =(),
                    parse_op = lambda x,**kw:  x,
                    )
        if 'normal' in self.data:
            format_db['normal'] = FormatEntry(
                pre = lambda x: np.array(x), 
                tftype  = tf.uint8,
                tfpre = lambda x: tf.io.encode_jpeg(x),
                featop = lambda x: _bytes_feature(x),
                feat_dtype = tf.string,
                feat_shape =(),
                parse_op = lambda x,**kw: tf.io.decode_image(x,channels=3,dtype=tf.float32) ,
            )
        
        if 'camera_info' in self.data:
            for k,shape in self.ci_format.items():
                if len(shape) == 1:
                    format_db[k] = FormatEntry(
                        pre = lambda x: x.flatten().tolist(), 
                        tftype  = tf.float32,
                        tfpre = lambda x: x,
                        featop = lambda x: _float_feature(x.flatten().tolist()),
                        feat_dtype = tf.float32,
                        feat_shape =shape,
                        parse_op = lambda x,**kw:  x,
                    )
                else:
                    shape1 = (functools.reduce(lambda x,y:x*y,shape),)
                    format_db[k] = FormatEntry(
                        pre = lambda x: x.flatten().tolist(), 
                        tftype  = tf.float32,
                        tfpre = lambda x: x,
                        featop = lambda x: _float_feature(x.flatten().tolist()),
                        feat_dtype = tf.float32,
                        feat_shape =shape1,
                        parse_op = lambda x,**kw:  x,
                    )
        return format_db

    def get_parser(self,):
        def datareader_parser(example_proto,format_db, extra_parse=None,extra_args={}):
            """Parses the example_proto Protobuf object and returns tf.Tensor 
            objects for the stored features
            
            example_proto : 
            format_db : dict
                Format specification to save the fitures
                Every key contains ->
                    pre_op # Preoperation
                    dtype  # dtype after pre_op
                    post_op # Post operation in TensorFlow
                    feature_op # Operation to transform in features
                    feat_dtype # Type after feature is created
                    feat_shape # Shape after feature is recovered
                    parse_op # Parse operation
            extra_parse : function
                Apply extra parse operations.
                Input as arguments : dictionary with the parsed tensors
                Return : a new dictionary with new parsed tensors
                Note the output of this function will be the only keeped dictionary.
            """
            features = {}
            for k in format_db.keys():
                #features[k] = tf.FixedLenFeature(format_db[k].feat_shape, format_db[k].feat_dtype)
                features[k] = format_db[k].feat_op(format_db[k].feat_shape, format_db[k].feat_dtype)


            data = tf.parse_single_example(example_proto, features,)
            #data = tf.parse_example(example_proto, features,)
            output_dict = {}
            args = {}
            if 'w' in format_db.keys() and 'h' in format_db.keys():
                args['h'] = tf.cast(format_db['h'].parse_op(data['h']),tf.int32)
                args['w'] = tf.cast(format_db['w'].parse_op(data['w']),tf.int32)
            for k in format_db.keys():
                if k in data:
                    var = format_db[k].parse_op(data[k],**args)
                else: 
                    print(k,'Not found... adding zero ')
                    var = tf.zeros(tf.float32,shape=(h,w,1))
                output_dict[k] = var
            if extra_parse is not None:
                output_dict = extra_parse(output_dict,**extra_args)
            return output_dict
        return datareader_parser
