import sys
import os
import json
import face_recognition
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
from cli import read_flags
from avgirls_struct import avgirls

def find_rectangle_of_face(image_name):
    def over_image_detector(image_shape, input, dim):
        if input < 0:
            return 0
        elif input > image_shape[dim]:
            return image_shape[dim]
        else:
            return input

    def bigger_image_with_10per(image_shape, top, right, bottom, left):
        len_x = abs(right - left)
        len_y = abs(bottom - top)
        len_x5 = int(len_x * 5 / 100)
        len_y5 = int(len_y * 5 / 100)
        max_face_loc['top']    = over_image_detector(image_shape, top    - len_y5, 1)
        max_face_loc['right']  = over_image_detector(image_shape, right  + len_x5, 0)
        max_face_loc['bottom'] = over_image_detector(image_shape, bottom + len_y5, 1)
        max_face_loc['left']   = over_image_detector(image_shape, left   - len_x5, 0)
        return max_face_loc

    image = face_recognition.load_image_file(image_name)
    face_locations = face_recognition.face_locations(image)
    max_face = 0
    max_face_loc = {}
    for face_location in face_locations:
        top, right, bottom, left = face_location
        if((bottom - top) * (right - left) > max_face):
            if max_face_loc:
                max_face_loc = {}
            max_face = (bottom - top) * (right - left)
            max_face_loc = bigger_image_with_10per(image.shape, top, right, bottom, left)
    return max_face_loc

def resize_image(image, output_dir, image_base_name, img_size, img_type):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    crop_resize_img = cv2.resize(image, (img_size, img_size))
    #cv2.imwrite(os.path.join(output_dir, image_base_name) , crop_resize_img)
    cv2.imencode(img_type, crop_resize_img)[1].tofile(os.path.join(output_dir, image_base_name))

def crop_resize_image(image_name, face_loc, output_dir, img_size, img_type, store_ = 1):
    image_base_name = os.path.basename(image_name)
    tlx = face_loc['left']
    tly = face_loc['top']
    brx = face_loc['right']
    bry = face_loc['bottom']
    print("Pixels of Face: {}".format(face_loc))
    img = cv2.imdecode(np.fromfile(image_name, dtype=np.uint8), cv2.IMREAD_COLOR)
    #img = cv2.imread(image_name, cv2.IMREAD_COLOR)
    crop_img = img[tly:bry, tlx:brx]
    try:
        if store_:
            resize_image(crop_img, output_dir, image_base_name, img_size, img_type)
        else:
            return cv2.resize(crop_img, (img_size, img_size))
    except cv2.error as e:
        print("having the {}, so passing it".format(e))

def storeEmbAsTFRecord(writer, emb, label):
    emb_str = emb.astype(np.float32).tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        'emb_str': tf.train.Feature(bytes_list=tf.train.BytesList(value=[emb_str]))
    }))
    writer.write(example.SerializeToString())

def read_and_decodeEmb(filename, batch_size, shape_):
    def parser(record):
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'emb_str' : tf.FixedLenFeature([], tf.string),
        }
        features = tf.parse_single_example(record, features)
        img = tf.decode_raw(features['emb_str'], tf.float32)
        img = tf.reshape(img, [shape_])
        #img = tf.cast(img, tf.float32)# * (1. / 255) - 0.5
        label = tf.cast(features['label'], tf.int32)
        return img, label
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(parser)
    #dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    #dataset = dataset.repeat(num_epochs)
    return dataset

def store_img_as_npy(output_dir, filename, vector_img):
    full_name = os.path.join(output_dir, filename + '.npy')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    np.save(full_name, vector_img)

def crop_img(ori_dir, crop_dir, img_size, img_type):
    ''' crop image form ori_dir and save it in crop_dir '''
    subfolders = [f.path for f in os.scandir(ori_dir) if f.is_dir() ]    
    if not os.path.isdir(crop_dir):
        os.makedirs(crop_dir)
    for sub_dir in subfolders:
        sub_dir_basename = os.path.basename(sub_dir)
        print("Crop step, Now is processing: {} ...".format(sub_dir_basename))
        onlyfiles = [f for f in os.listdir(sub_dir) if os.path.isfile(os.path.join(sub_dir, f))]
        for f in onlyfiles:
            image_name = os.path.join(sub_dir, f)    
            print(image_name)    
            face_loc = find_rectangle_of_face(image_name)
            if not face_loc:
                print("Not found the Face in the Picture")
            else:
                crop_resize_image(image_name, face_loc, os.path.join(crop_dir, sub_dir_basename), img_size, img_type)

def creat_npy(crop_dir, npy_dir, model):
    ''' create npy file for tfrecord '''
    from facenet.src.facenet import load_model, prewhiten
    with tf.Graph().as_default():
        with tf.Session() as sess:
            print("Now loading the model...")
            load_model(model)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            subfolders = [f.path for f in os.scandir(crop_dir) if f.is_dir() ]
            for sub_dir in subfolders:
                count = 0
                sub_dir_basename = os.path.basename(sub_dir)
                print("CreatNpy step, Now is processing: {} ...".format(sub_dir_basename))
                onlyfiles = [f for f in os.listdir(sub_dir) if os.path.isfile(os.path.join(sub_dir, f))]
                for f in onlyfiles:
                    image_name = os.path.join(sub_dir, f)
                    print(image_name)
                    try:
                        img = cv2.imdecode(np.fromfile(os.path.expanduser(image_name),dtype=np.uint8), cv2.IMREAD_COLOR)
                        prewhitened = prewhiten(img)
                        feed_dict = { images_placeholder: [prewhitened], phase_train_placeholder:False }
                        emb = sess.run(embeddings, feed_dict=feed_dict)
                        store_img_as_npy(os.path.join(npy_dir, sub_dir_basename), sub_dir_basename + '_' + str(count), emb)
                        count = count + 1
                    except TypeError as e:
                        print("having the {}, so passing it".format(e))
                        
def create_tfrecord(npy_dir, model, save_json, tfRecord, av):
    ''' create tfrecord from npy for inferencing '''
    writer = tf.python_io.TFRecordWriter(tfRecord)
    subfolders = [f.path for f in os.scandir(npy_dir) if f.is_dir() ]
    for sub_dir in subfolders:
        sub_dir_basename = os.path.basename(sub_dir)
        print("TFstep, Now is processing: {} ...".format(sub_dir_basename))       
        onlyfiles = [f for f in os.listdir(sub_dir) if os.path.isfile(os.path.join(sub_dir, f))]
        for f in onlyfiles:
            filename = os.path.join(sub_dir, f)
            print("now is ", av.dict_name[sub_dir_basename], f )
            emb = np.load(filename)
            storeEmbAsTFRecord(writer, emb, av.dict_name[sub_dir_basename])
    if save_json:
        with open(os.path.join(npy_dir, av.json_file), 'w') as fp:
            json.dump(av.list_name, fp)
    writer.close()


def inference(model, tfRecord, img_paths, top, av):
    # step1 crop the face in the list
    img_list = []
    for image_name in img_paths:
        face_loc = find_rectangle_of_face(image_name)
        if not face_loc:
            print("Not found the Face in the Picture")
        else:
            img_list.append(crop_resize_image(image_name, face_loc, "", av.img_size, av.img_type, store_ = 0))
    # step 2 store img as the np format
    from facenet.src.facenet import load_model, prewhiten
    emb_list = []
    for i in img_list:
        with tf.Graph().as_default():
            with tf.Session() as sess:
                load_model(model)
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                prewhitened = prewhiten(i)
                feed_dict = { images_placeholder: [prewhitened], phase_train_placeholder:False }
                emb = sess.run(embeddings, feed_dict=feed_dict)
                emb_list.append(emb)
    
    dataset = read_and_decodeEmb(tfRecord, av.batch_size, 512) # 512 is face vector of the model
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    init_op = tf.group(tf.global_variables_initializer(),
        tf.local_variables_initializer())    
    # step 3 calculate the confidence level
    dist_dict  = {}
    label_dict = {}
    with tf.Session()  as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for emb in emb_list:
            sess.run(iterator.initializer)
            while True:
                try:
                    next_example, next_label = sess.run(next_element)
                    for i, ele in enumerate(next_example):
                        if str(next_label[i]) not in dist_dict:
                            dist_dict[str(next_label[i])]  = 0
                        if str(next_label[i]) not in label_dict:
                            label_dict[str(next_label[i])] = 0
                        dist = np.sqrt(np.sum(np.square(np.subtract(ele, emb))))
                        dist_dict[str(next_label[i])]  = dist + dist_dict[str(next_label[i])] 
                        label_dict[str(next_label[i])] =    1 + label_dict[str(next_label[i])]
                except tf.errors.OutOfRangeError:
                    break
        coord.request_stop()
        coord.join(threads)

    for i in range(len(av.list_name)):
        dist_dict[str(i)] = (2 * label_dict[str(i)] - dist_dict[str(i)]) * 50 / label_dict[str(i)]
    from collections import Counter 
    k = Counter(dist_dict) 
    top = av.list_name if top > len(av.list_name) else top
    high = k.most_common(top) 
    for i, ele in enumerate(high):
        print("第 {} 相似為: {}, 相似度: {:.1f}%".format(i + 1, av.rdict_name[ele[0]], ele[1]))

def main(flags):
    av = avgirls()
    av.updata_parameter_from_flags(flags)
   
    if flags.only_crop:
        crop_img(flags.ori_dir, flags.crop_dir, av.img_size, av.img_type)
    elif flags.only_npy:
        creat_npy(flags.crop_dir, flags.npy_dir, flags.model_path)    
    elif flags.only_svae_tfrecord:
        create_tfrecord(flags.npy_dir, flags.model_path, flags.save_json, flags.tfrecord_path, av)
    elif flags.preprocess:
        crop_img(flags.ori_dir, flags.crop_dir, av.img_size, av.img_type)
        creat_npy(flags.crop_dir, flags.npy_dir, flags.model_path)    
        create_tfrecord(flags.npy_dir, flags.model_path, flags.save_json, flags.tfrecord_path, av)
    elif flags.inference:
        inference(flags.model_path, flags.tfrecord_path, flags.image_path, flags.top, av)

if __name__ == '__main__': 
    flags = read_flags()
    main(flags)