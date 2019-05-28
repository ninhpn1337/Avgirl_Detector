# Avgirl_Detector
This tool is used to detect the most likely japan av girl in your pictures.
(Now only support python3)

## Step 1, Download the neccesary file form git can goolge drive:

$ python3 setup.py

## Step 2, Make sure that you've installed neccesary module at requirements.txt

$ pip3 install -r requirements.txt 

## Step 3 Do inference:

$ python3 avgirls_detector.py -inf 1 -i "/your/file/full/path/name"

### Do  multiple inferencing with same people in the pictures:

$ python3 avgirls_detector.py -inf 1 -i "/your/file/full/path/name1" "/your/file/full/path/name2" 

### for example:
$ python3 avgirls_detector.py -inf 1 -i ${HOME}/downloads/img1.png ${HOME}/downloads/img2.png

## Check another tool methods:

$ python3 avgirls_detector.py -h

## You can create your pictures into my database,ant the easiest way is:

$ mkdir data/ori 

$ mkdir data/ori/"sub_dir_of_your_pic"

$ cp "all_your_pictures" data/ori/"sub_dir_of_your_pic" 

$ mkdir data/crop

$ python3 avgirls_detector.py -pp 1



