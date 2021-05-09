import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from glob import glob

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
model_path = "./"
PATH_TO_CKPT = model_path + MODEL_NAME + '/frozen_inference_graph.pb'


def download_model():
    import six.moves.urllib as urllib
    import tarfile

    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())
            
            
#Instantiates a tensorflow "detection" graph
def load_graph():
    if not os.path.exists(PATH_TO_CKPT):
        download_model()

    #   From the tf library (as_default method):
    
    #   "Returns a context manager that makes this Graph the default graph.
    #   This method should be used if you want to create multiple graphs in the same process. 
    #   For convenience, a global default graph is provided, and all ops will be added to this graph 
    #   if you do not create a new graph explicitly."
    
    #   "with" statement: ensures proper acquisition and release of resources
    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            # Creation of tf graph from file 
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    return detection_graph

# In each box there will be a detected object 

def select_boxes(boxes, classes, scores, score_threshold=0, target_class=10):
    """
    :param boxes:
    :param classes:
    :param scores:
    :param target_class: default traffic light id in COCO dataset is 10
    :return:
    """
    
    #   From the numpy library (np.squeeze):
    #   The input array, but with all or a subset of the dimensions of length 1 removed. 
    #   This is always a itself or a view into a.
    
    sq_scores = np.squeeze(scores)
    sq_classes = np.squeeze(classes)
    sq_boxes = np.squeeze(boxes)

    sel_id = np.logical_and(sq_classes == target_class, sq_scores > score_threshold)
    
    return sq_boxes[sel_id]