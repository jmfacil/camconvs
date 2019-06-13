import numpy as np
from abc import ABC, abstractmethod

class Generator:
    def __init__(self):
        pass
    @abstractmethod
    def __call__(self):
        """
        Yields a dictionary with the next sample.
        """
        raise NotImplementedError("The method __call__ of classes that inherit from Generator must be implemented.")
    @abstractmethod
    def number_of_examples(self):
        raise NotImplementedError("The method number_of_examples of classes that inherit from Generator must be implemented.")
    @abstractmethod
    def format_entry(self):
        if hasattr(self,format_json):
            raise NotImplementedError("The format entry method from json is not yet implemented")
        else:
            raise NotImplementedError("The format entry must be implemented")

    @staticmethod
    def read_depth(filename):
        # loads depth map D from png file
        # and returns it as a numpy array,
        # for details see readme.txt

        depth_png = np.array(Image.open(filename), dtype=int)
        # make sure we have a proper 16bit depth map here.. not 8bit!

        depth = depth_png.astype(np.float) / 256.
        depth[depth_png == 0] = np.NaN
        return depth
    
    @staticmethod
    def read_float(name):
        f = open(name, 'rb')
        if(f.readline().decode("utf-8"))  != 'float\n':
            raise Exception('float file %s did not contain <float> keyword' % name)
        dim = int(f.readline())
        dims = []
        count = 1
        for i in range(0, dim):
            d = int(f.readline())
            dims.append(d)
            count *= d
        dims = list(reversed(dims))
        data = np.fromfile(f, np.float32, count).reshape(dims)
        if dim > 2:
            data = np.transpose(data, (2, 1, 0))
            data = np.transpose(data, (1, 0, 2))
        return data

    @staticmethod
    def read_rgb_pil(filename):
        pilimg = PILImage.open(filename)
        pilimg.load()
        if pilimg.mode != 'RGB':
            pilimg = pilimg.convert('RGB')
        return pilimg

    @staticmethod
    def read_rgb(filename):
        return np.asarray(read_rgb_pil(filename))
