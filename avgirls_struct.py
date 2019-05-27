import os
import sys
import json
import pickle

class avgirls(object):
    def __init__(self):
        self.list_name = []
        self.dict_name = {}
        self.rdict_name = {}
        self.dict_pickle = {}
        self.img_size = 160
        self.batch_size = 32
        self.img_type = '.png'
        self.json_file = 'data.json'

    def read_list_from_json(self, npy_dir):
        with open(os.path.join(npy_dir, self.json_file), 'r') as fp:
            self.list_name = json.load(fp)
            self.creat_dict_from_list()
    
    def read_list_from_npy_dir(self, npy_dir):
        self.list_name = [dI for dI in os.listdir(npy_dir) if os.path.isdir(os.path.join(npy_dir, dI))]
        self.creat_dict_from_list()

    def creat_dict_from_list(self):
        for i, ele in enumerate(self.list_name):
            self.dict_name[ele] = i 
        self.rdict_name = {str(v) : k for k, v in self.dict_name.items()}
    
    def updata_parameter_from_flags(self, flags):
        if flags.read_json:
            self.read_list_from_json(flags.npy_dir)
        else:
            self.read_list_from_npy_dir(flags.npy_dir)
        self.batch_size = flags.batch_size
        tmp = (flags.img_type).split('.')
        self.img_type  = '.' + tmp[-1] if len(tmp) > 1 else '.' + flags.img_type
        