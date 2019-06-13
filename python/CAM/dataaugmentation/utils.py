
import json

def transform_bools(d):
    """
    Transform all the string with value "True" or "False" by its respective
    boolean value in a dictionary, it also applies it with neasted Dictionaries.

    d: dict 
        the dictionary where the transformation will take place
           """
    for k,v in d.items():
            if isinstance(v,dict):
                    d[k].update(transform_bools(d[k]))
            else:
                    if v == "True":
                            v = True
                    elif v == "False":
                            v = False
                    d[k]=v
    return d

class CAMAugmentSettings():
    """
    Settings for camera data augmentation.

    json_file : string
        file name of the settings file
         Example of file:
        {
          "specific": [
            {
              "bs": 3,
              "kargs": {
                "target_w": 320,
                "target_h": 320,
                "debug": "True"
              }
            },
            {
              "bs": 1,
              "file":"anotherfile.tfrecord",
              "args": {
                "target_w": 480,
                "target_h": 160,
                "depth_keys": [
                    "depth1",
                    ]
              }
            },
            {
              "bs": 3,
              "args": {
                "target_w": 256,
                "target_h": 256,
                "debug": "True"
              }
            }
          ],
          "default": {
            "dataoperations": [
              "normalize_images",
              "principal_point_random_shift",
              "focal_aug_by_cropping",
              "rgb_augmentation",
              "h_mirror",
              "generate_groundtruth"
            ],
            "args": {
              "from_inverse_depth":"False",
              "to_inverse_depth":"True",
              "focal_norm":"False",
              "focal_factor":500.0,
              "pp_max_shift":100,
              "data_format": "channels_last",
              "rgb_aug": {
                "image": {
                  "hue": 0.05555555,
                  "lower_saturation": 0.8,
                  "upper_saturation": 1.2,
                  "value": 0.1,
                  "bright": 0.08,
                  "lower_contrast": 0.9,
                  "upper_contrast": 1.1,
                  "gamma": 0.2
                }
              },
              "debug": "False",
              "image_keys": ["image"],
              "depth_keys": [
                "depth"
              ],
              "normal_keys":["normal"],
              "h_mirror_th": 0.5
            },
            "file" : ["dataset1.tfrecord","dataset2.tfrecord"]
            ]
          }
        }
    """
    class SpecificSettings():
        def __parse_args__(self,dop,args):
            switcher = {
                    "normalize_images": self.normalize_images,
                    "principal_point_random_shift": self.principal_point_random_shift,
                    "focal_aug_by_cropping":
                    }
        def __init__(self,dic,extra_module):
            self.extra_module = extra_module
            self.dataoperations = []
            self.args = {}
            for dop in dic["dataoperations"]:
                _args=self.__parse_args__(dop,dic['args'])

            if "normalize_images" in dic
            
            
    def __init__(self,json_file,extra_module = None):
        self.parameters=[]
        with open(json_file) as f:
            data = json.load(f)
            data=transform_bools(data)
            for d in data['specific']:
                params={}
                params.update(deepcopy(data['default']))
                for k,v in d.items():
                    if isinstance(v,dict):
                        params[k].update(v)
                    else:
                        params[k]=v
                self.parameters.append(params)
    def 

