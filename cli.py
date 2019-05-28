import argparse
import os

def read_flags():
    """Returns flags"""
    parser = argparse.ArgumentParser(
        description = 'This is an av girls detector, please make sure using the correct arugment!!', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument( 
        "-b", "--batch_size", default=8, 
        type=int, help="batch_size for loading data from tfrecorder")

    parser.add_argument( 
        "-d", "--data_dir", default=os.path.join(os.getcwd(), "data"), 
        type=str, help="put ori_dir, crop_dir, npy_dir in this dir")

    parser.add_argument( 
        "-m", "--model_path", default=os.path.join(os.getcwd(), "models", "20180402-114759"), 
        type=str, help="the model for prcessing the img")
    
    parser.add_argument(
        '-i', '--image_path', nargs='+', default = [], type=str, help="image files for inferenced")
        
    parser.add_argument(
        '-it', '--img_type', default='png', type=str, help="the crop img data type")

    parser.add_argument( 
        "-t", "--tfrecord_path", default=os.path.join(os.getcwd(), "av.tfrecords"), 
        type=str, help="the tfrecord full path")

    parser.add_argument( 
        "-od", "--ori_dir", default=os.path.join(os.getcwd(), "data", "ori"), 
        type=str, help="put original img this dir, make sure different class img should store in sub_dir")

    parser.add_argument( 
        "-cd", "--crop_dir", default=os.path.join(os.getcwd(), "data", "crop"), 
        type=str, help="the detector put face img to this dir")

    parser.add_argument( 
        "-nd", "--npy_dir", default=os.path.join(os.getcwd(), "data", "npy"),
         type=str, help="the necessary npy dir for inferenced the client img")

    parser.add_argument( 
        "-rj", "--read_json", default=0,
         type=int, help="read av girls list from the json file")

    parser.add_argument( 
        "-sj", "--save_json", default=0,
         type=int, help="save av girls list to the json file")
    
    parser.add_argument(
        '-top', '--top', default = 5, type=int, help="number of av girls that img are most like")

    group = parser.add_mutually_exclusive_group()
    
    group.add_argument(
        "-oc", "--only_crop", default = 0, type=int, help="only crop the image to crop_dir")
    
    group.add_argument(
        "-on", "--only_npy", default = 0,  type=int, help="only save crop image form crop_dir as npy file in npy_dir")
    
    group.add_argument(
        "-ost", "--only_svae_tfrecord", default = 0,  type=int, help="only save npy file as tfrecord")
    
    group.add_argument(
        "-pp", "--preprocess", default = 0,  type=int, help="preprocess: doing crop, trans2npy, save npy as tfrecord")
    
    group.add_argument(
        "-inf", "--inference", default = 0,  type=int, help="do inferencing the client imgs")
    
    flags = parser.parse_args()
    return flags