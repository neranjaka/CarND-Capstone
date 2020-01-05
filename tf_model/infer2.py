import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from scipy.stats import norm

SSD_GRAPH_FILE = 'frozen_models/frozen_sim_mobile/frozen_inference_graph.pb'

tl = {'1': 'Green', '2': 'Red', '3': 'Yellow' , '4' : 'OFF'  }
class inference():

    def init(self):
        """Loads a frozen inference graph"""
        self.confidence_cutoff = None
        self.graph= None
        self.image_tensor = None
        self.detection_boxes = None
        self.detection_scores = None
        self.detection_classes = None
    
    def init2(self, graph_file= SSD_GRAPH_FILE):
        self.graph = self.load_graph(graph_file)
        self.init_tensors()
        self.confidence_cutoff = 0.8
        
    def init_tensors(self):
        self.graph = self.load_graph()
        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.graph.get_tensor_by_name('detection_scores:0')
        # The classification of the object (integer id).
        self.detection_classes = self.graph.get_tensor_by_name('detection_classes:0')
    

    def load_graph(self, graph_file = SSD_GRAPH_FILE):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph

    def filter_boxes(self, min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)
        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    def to_image_coords(self, boxes, height, width):
        """
        The original box coordinate output is normalized, i.e [0, 1].
        This converts it back to the original coordinate based on the image
        size.
        """
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width
        return box_coords

    def infer(self, image = 'test_images/left0560.jpg'):
        image = Image.open(image)
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
        with tf.Session(graph=self.graph) as sess:                
            # Actual detection.
            (boxes, scores, classes) = sess.run([self.detection_boxes, self.detection_scores, self.detection_classes], 
                                                feed_dict={self.image_tensor: image_np})
            # Remove unnecessary dimensions
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)
            # Filter boxes with a confidence score less than `confidence_cutoff`
            boxes, scores, classes = self.filter_boxes(self.confidence_cutoff, boxes, scores, classes)
            # The current box coordinates are normalized to a range between 0 and 1.
            # This converts the coordinates actual location on the image.
            width, height = image.size
            box_coords = self.to_image_coords(boxes, height, width)
            print(scores,[tl[str(int(i))] for i in classes])

i  = inference()
i.init2()
for img in glob.glob("test_images/*.jpg"):
    print(img)
    t0 = time.time()
    i.infer(img)
    print(time.time() - t0)
