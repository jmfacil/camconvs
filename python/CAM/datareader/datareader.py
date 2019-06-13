import tensorflow as tf 
from collections import namedtuple
from .utils import ProgressBar, MessageWaiting
from pathlib import Path,PureWindowsPath

#FormatEntry = namedtuple('FormatEntry',['pre_op','dtype','post_op','feature_op','feat_dtype','feat_shape','parse_op'])
#FormatEntry = namedtuple('FormatEntry',['pre','dtype','post','feature','feat_dtype','feat_shape','parse_op'])
#FormatEntry = namedtuple('FormatEntry',['pre','tftype','tfpre','featop','dtype','post','feature','feat_dtype','feat_shape','parse_op'])

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value if isinstance(value, (list,tuple)) else [value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value if isinstance(value, (list,tuple)) else [value]))


_fields_=['pre','tftype','tfpre','featop','feat_dtype','feat_shape','feat_op','parse_op']
_defaults_=[lambda x:x,tf.float32,lambda x:x,lambda x:_float_feature(x),tf.float32,(),tf.FixedLenFeature,lambda x:x]
FormatEntry = namedtuple('FormatEntry',_fields_)
FormatEntry.__new__.__defaults__=tuple(_defaults_)

class Datawriter():

    """
    """
    def __init__(self,filename,generator,max_examples,format_entry,compression_type = tf.python_io.TFRecordCompressionType.GZIP,debug=False):
        self.debug_msg = "[DEBUG](datareader.Datawriter): "
        self.max_examples = max_examples
        self.debug = debug
        self.filename = filename
        path = Path(self.filename.replace('/*.tfrecord',''))
        path.parents[0].mkdir(exist_ok = True, parents=True)
        self.generator = generator
        self.format_entry = format_entry
        self.compression_type = compression_type
        self.progress_bar = ProgressBar(msg='Creating '+self.filename,
                waiting_char='█', length = 23,end_msg = 'COMPLETED',total=self.max_examples)
        self.done = False
    def write(self,):
        if self.done:
            return
        if not self.debug:
            self.progress_bar.start()
        gpu_options = tf.GPUOptions()
        # dont use the gpu
        session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options, device_count = {'GPU': 0}))
        examples_count = 0
        
        # Prepare tensorflow preprocessing
        pretf = {}
        posttf = {}
        for k in self.format_entry:
            if self.debug:
                print(self.debug_msg, 'FormatEntry for key: ',k,self.format_entry[k])
            pretf[k]=tf.placeholder(self.format_entry[k].tftype)
            posttf[k] = self.format_entry[k].tfpre(pretf[k])



        options = tf.python_io.TFRecordOptions(compression_type =self.compression_type )

        # Load from generator apply pre and tfpre processing
        with tf.python_io.TFRecordWriter(self.filename,options) as writer:
            view = next(self.generator())
            while view is not None and examples_count <= self.max_examples:
                feed_dict={}
                for k in view.keys():
                    feed_dict[pretf[k]] = self.format_entry[k].pre(view[k])
                
                posttfdata = session.run(posttf,feed_dict)
    
                feature = {}
                for k in view.keys():
                    feature[k] = self.format_entry[k].featop(posttfdata[k])

                features = tf.train.Features(feature=feature)
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())
                
                examples_count += 1
                if not self.debug:
                    self.progress_bar.update(examples_count)
                else:
                    print(self.debug_msg, 'Examples writen: ',examples_count)
                view = next(self.generator())
        if not self.debug:
            self.progress_bar.finish_wait()
        else:
            print(self.debug_msg, 'DONE')
        self.done = True
        del session

class Datareader():

    """

    """
    def __init__(self,files,batch_size=1,threads=1,parser = lambda x: x,buffer_size=100,one_shot=True,shuffle =False,debug=False,
            compression_type = 'GZIP',name = 'datareader'):
        if not isinstance(files,list):
            self.files = [files]
        else:
            self.files = files

        self.batch_size = batch_size
        self.threads = threads
        self.parser = parser
        self.buffer_size = buffer_size
        self.one_shot = one_shot
        self.shuffle = shuffle
        self.debug = debug
        self.debug_msg = "[DEBUG](datareader.Datareader): "

        with tf.name_scope(name):
            if self.debug:
                print(self.debug_msg,'Creating datareader for files: ',self.files)
            infinite_record_dataset = tf.data.TFRecordDataset(files,compression_type=compression_type).repeat() 
            if self.debug:       
                print(self.debug_msg,'Record dataset created. ')
            parsed_record = infinite_record_dataset.map(self.parser, num_parallel_calls=max(1,self.threads))
            if self.debug:       
                print(self.debug_msg,'Record dataset parsed.')

            if self.shuffle:
                sparsed_record=parsed_record.shuffle(buffer_size=self.buffer_size)
                if self.debug:       
                    print(self.debug_msg,'Record dataset shuffled.')
            else:
                sparsed_record=parsed_record

            #prefetched_data = sparsed_record.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size)).prefetch(self.buffer_size)
            prefetched_data = sparsed_record.batch(self.batch_size,drop_remainder = True).prefetch(self.buffer_size)

            if self.debug:       
                print(self.debug_msg,'Prefeched data...')
                print(self.debug_msg,'One shot', one_shot)
            if one_shot:
                iterator = prefetched_data.make_one_shot_iterator()
                next_element = iterator.get_next()
                self.datareader = next_element
            else:
                print('[WARNING](datareader.Datareader) Datareader has not been fully-tested without one_shot option set to True')
                print('                                 Use only under your responsability')
                iterator = prefetched_data.make_initializable_iterator()
                self.datareader = next_element,iterator

    def __call__(self,):
        return self.datareader
        
