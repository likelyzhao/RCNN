from __future__ import print_function
import cPickle
import cv2
import os
import numpy as np

from imdb import IMDB
from imagenet_eval import imagenet_eval, imagenet_eval_detailed, draw_ap, draw_map
from ds_utils import unique_boxes, filter_small_boxes


class wilderface(IMDB):
    def __init__(self, image_set, root_path, devkit_path):
        """
        fill basic information to initialize imdb
        :param image_set: 2007_trainval, 2007_test, etc
        :param root_path: 'selective_search_data' and 'cache'
        :param devkit_path: data and results
        :return: imdb object
        """
        # year, image_set = image_set.split('_')
        super(wilderface, self).__init__('wilderface_', image_set, root_path, annotationfilepath)  # set self.name
        # self.year = year
#    print (devkit_path)
#    print ("devkit")
        self.root_path = root_path
        self.devkit_path = devkit_path
        self.data_path = os.path.join(devkit_path, 'DET')

        self.classes = imagenet_classes 
        self.num_classes = len(self.classes)
        self.image_set_filepath = annotationfilepath
        self.num_images = len(self.image_set_index)
        print('num_images', self.num_images)
"""
0--Parade/0_Parade_marchingband_1_849.jpg
1
449 330 122 149 0 0 0 0 0 0
"""

        self.config = {'comp_id': 'comp4',
                       'use_diff': False,
                       'min_size': 2}

    def _load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or val)
        :return:
        """
        image_set_index_file = os.path.join(self.data_path, 'ImageSets', 'DET', self.image_set + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file) as f:
            if self.image_set == "val":
                image_set_index = [x.strip().split(' ')[0] for x in f.readlines()]
            elif self.image_set == "train":
                image_set_index = [x.strip().split(' ')[0] for x in f.readlines()]
            else:
                image_set_index = [x.strip() for x in f.readlines()]
        return image_set_index

    def image_path_from_index(self, index):
        """
        given image index, find out full path
        :param index: index of a specific image
        :return: full path of this image
        """
        image_file = os.path.join(self.data_path,'Data','DET', self.image_set, index + '.JPEG')
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file
    def _code_to_label(code):
        assert len(code) = 

    def _load_db_imagenset_file(filepath):
        gt_roidb =[]
        f = open(filepath)
        while 1 :
            roi_rec = dict()
            line = f.readline()
            if line is None:
               break
             roi_rec['image'] = os.path.join(self.rootpath,line)
             size = cv2.imread(roi_rec['image']).shape
             roi_rec['height'] = size[0]
             roi_rec['width'] = size[1] 
             num_face =int( f.readline())

             boxes = np.zeros((num_face, 4), dtype=np.uint16)
             gt_classes = np.zeros((num_face), dtype=np.int32)
             overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # Load object bounding boxes into a data frame.
             for ix  in range(num_face):
                 bboxline = f.readline()
         #        449 330 122 149 0 0 0 0 0 0
                 bbox = bboxline.split(' ')[0:4]
                 # Make pixel indexes 0-based
                 x1 = float(bbox[0])
                 y1 = float(bbox[1])
                 x2 = float(bbox[0]) + float(bbox[2])
                 if x2 == size[1]:
                     print ("label xmax reach the image width")
                     x2 = x2 - 1
                 y2 = float(bbox[1])+ float(bbox[3])
                 if y2 == size[0]:
                     print ("label ymax reach the image height")
                     y2 = y2 - 1
             cls = class_to_index[obj.find('name').text.lower().strip()]
             boxes[ix, :] = [x1, y1, x2, y2]
             gt_classes[ix] = cls
             overlaps[ix, cls] = 1.0


    def gt_roidb(self):
        """
        return ground truth image regions database
        :return: imdb[image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            for gt in roidb:
                if gt['boxes'].shape[0]==0:
                    print(gt['image'])
                return roidb
        
        gt_roidb = _load_db_imageset_file(self.image_set_filepath)
  
        gt_roidb = [self.load_imagenet_annotation(index) for index in self.image_set_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def load_imagenet_annotation(self, index):
        """
        for a given index, load image and bounding boxes info from XML file
        :param index: index of a specific image
        :return: record['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """

        import xml.etree.ElementTree as ET
        roi_rec = dict()
        roi_rec['image'] = self.image_path_from_index(index)
        size = cv2.imread(roi_rec['image']).shape
        roi_rec['height'] = size[0]
        roi_rec['width'] = size[1]

        filename = os.path.join(self.data_path, 'Annotations','DET',self.image_set, index + '.xml')
#    print (filename)
        tree = ET.parse(filename)
    #print(tree)
        objs = tree.findall('object')
#        if not self.config['use_diff']:
 #           non_diff_objs = [obj for obj in objs if int(obj.find('difficult').text) == 0]
 #           objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        class_to_index = dict(zip(self.classes, range(self.num_classes)))
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) 
            y1 = float(bbox.find('ymin').text) 
            x2 = float(bbox.find('xmax').text)
            if x2 == size[1]:
                print ("label xmax reach the image width")
                x2 = x2 - 1
            y2 = float(bbox.find('ymax').text)
            if y2 == size[0]:
                print ("label ymax reach the image height")
                y2 = y2 - 1
            cls = class_to_index[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        roi_rec.update({'boxes': boxes,
                        'gt_classes': gt_classes,
                        #'gt_overlaps': overlaps,
                        'max_classes': overlaps.argmax(axis=1),
                        'max_overlaps': overlaps.max(axis=1),
                        'flipped': False})
        return roi_rec

