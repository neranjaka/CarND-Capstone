from glob import glob
import tensorflow as tf
import numpy as np
import cv2

TRAFFIC_LIGHTS = ['Green', 'Red', 'Yellow', 'Unknown']

class TLClassifier(object):
    def __init__(self, is_site):
        if is_site == True:
            model = 'frozen_models/site/frozen_inference_graph.pb'
        elif is_site == False:
            model = 'frozen_models/simulator/frozen_inference_graph.pb'
        self.detection_graph = self.load_graph(model)

        self.image_tensor, self.detection_boxes, \
        self.detection_scores, self.detection_classes = self.extract_tensors()
        self.sess = tf.Session(graph = self.detection_graph)

    def load_graph(self, graph_file):
        """ Loads frozen inference graph, the pretrained model """
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph

    def extract_tensors(self):
        """ Extract relevant tensors for detecting objects """
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        return image_tensor, detection_boxes, detection_scores, detection_classes

    def filter_boxes(self, min_score, boxes, scores, classes):
        """ Return boxes with a confidence >= `min_score` """
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)
        
        filtered_boxes = boxes[idxs,  ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_np = np.expand_dims( np.asarray(image_rgb, dtype=np.uint8), 0)

        with tf.Session(graph = self.detection_graph) as sess:
            (boxes, scores, classes) = sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
                                                 feed_dict={self.image_tensor: image_np})

            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)

            confidence_cutoff = 0.6
            boxes, scores, classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes)
        if len(scores) <= 0:
            traffic_light_class_id = 4
            # print("Traffic Light UNKNOWN")
            traffic_light_state = "UNKNOWN"
            # print("traffic_light_state = %s", traffic_light_state)
            return traffic_light_state

        # traffic light detected, return light state for green, yellow, red classification
        traffic_light_class_id = int(classes[np.argmax(scores)])
        # print("traffic light class id = %s", traffic_light_class_id)

        if traffic_light_class_id == 1:
            # print("Traffic Light GREEN")
            traffic_light_state = "GREEN"
        elif traffic_light_class_id == 2:
            # print("Traffic Light RED")
            traffic_light_state = "RED"
        elif traffic_light_class_id == 3:
            # print("Traffic Light YELLOW")
            traffic_light_state = "YELLOW"

        # print("traffic_light_state = %s", traffic_light_state)
        return traffic_light_state


def test(images_root = 'training_images_simulator'):
    total_images = 0
    c = TLClassifier(False)
    print("procesing images and classifying ...")
    for fldr in glob(images_root+"/*"):
        print(fldr)
        good_classification = 0
        bad_classification = 0
        expected = fldr.split('/')[-1].upper()
        for imgpath in glob(fldr+ "/*.jpg"):
            print(imgpath)
            img = cv2.imread(imgpath,0)
            prediction = c.get_classification(img)
            if expected != prediction: 
                bad_classification += 1
                print(imgpath, expected, prediction )
            else:
                good_classification += 1
        print("FOLDER = {} , Good = {} , Bad = {}".format(expected,good_classification,bad_classification)) 

if __name__ == "__main__":
    test()