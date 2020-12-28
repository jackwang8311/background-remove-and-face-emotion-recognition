import numpy as np
import tensorflow as tf
import transfer

v_tf = tf.__version__.split('.')[0]
if v_tf == '2':
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
#import imutils

import torch
from net import Net


import os
from stuff.helper import FPS2, WebcamVideoStream
#from skimage import measure
from random import randint


from tensorflow.keras.preprocessing.image import img_to_array
import imutils
import cv2
from tensorflow.keras.models import load_model
import numpy as np

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)


EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]




img_num = 0
mike_flag = True




def load_model1():
    print('Loading model...')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        seg_graph_def = tf.GraphDef()
        with tf.gfile.GFile('models/deeplabv3_mnv2_pascal_train_aug/frozen_inference_graph.pb', 'rb') as fid:
            serialized_graph = fid.read()
            seg_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(seg_graph_def, name='')
    return detection_graph

def loadrecongmodel(path):
    emotion_classifier = load_model(path, compile=False)
    graph = tf.get_default_graph()

    return graph,emotion_classifier


def predict(faces,gray,graph,model):
    """
    图片文字方向预测
    """
    #emotion_classifier = load_model(emotion_model_path, compile=False)
    #graph = tf.get_default_graph()
    with graph.as_default():
         if len(faces) > 0:
            print('ddddd')
            faces = sorted(faces, reverse=True,
                           key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces
            # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
            # the ROI for classification via the CNN
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = model.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]
            return label
         else:
             print('ddddd22')
             return "222"




def next_bg(event, x, y, flags, param):
    global mike_flag, img_num
    if event == cv2.EVENT_LBUTTONUP:
        mike_flag = not mike_flag
    elif event == cv2.EVENT_RBUTTONUP:
        img_num += 1


def segmentation(detection_graph):
    vs = WebcamVideoStream(0, 640, 480).start()
   # vs1 = WebcamVideoStream(0, 1280, 720).start()
   # cv2.namedWindow('your_face')

    resize_ratio = 1.0 * 513 / max(vs.real_width, vs.real_height)
    target_size = (int(resize_ratio * vs.real_width),
                   int(resize_ratio * vs.real_height))
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    fps = FPS2(5).start()

    filelist = [file for file in os.listdir(
        'backgrounds') if file.endswith('.jpg')]

    num_files = len(filelist)

    background_image = []
    resized_background_image = []
    for x in filelist:
        background_image.append(cv2.imread(x))
        resized_background_image.append(cv2.resize(
            background_image[-1], target_size))

    style_model = Net(ngf=128)
    model_dict = torch.load('models/21styles.model')
    model_dict_clone = model_dict.copy()
    for key, value in model_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_dict[key]
    style_model.load_state_dict(model_dict, False)

    # fff = 0

    # background_image = cv2.imread('b.jpg')
    # resized_background_image = cv2.resize(
    #     background_image, target_size)  # (384,513)

    # background_image2 = cv2.imread('b2.jpg')
    # resized_background_image2 = cv2.resize(
    #     background_image2, target_size)  # (384,513)

    # background_image3 = cv2.imread('b3.jpg')
    # resized_background_image3 = cv2.resize(
    #     background_image3, target_size)  # (384,513)



    # Uncomment to save output
    # out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc(
    # 'M', 'J', 'P', 'G'), 1, (vs.real_height, vs.real_width))#CHANGE

    print("Starting...")

    cv2.namedWindow('segmentation', 16)  # 16 means WINDOW_GUI_NORMAL, to disable right click context menu

    cv2.setMouseCallback('segmentation', next_bg)

    #cv2.namedWindow('segmentation', 16)  # 16 means WINDOW_GUI_NORMAL, to disable right click context menu

   # cv2.setMouseCallback('segmentation', next_bg)
    graph1,model1=loadrecongmodel(emotion_model_path)

    img_num = 0
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
             while vs.isActive():
                      image = imutils.resize(vs.read(), width=300)

                      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                      faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
                      a = predict(faces, gray,graph1,model1)
                      #print(a)
         # canvas = np.zeros((250, 300, 3), dtype="uint8")
         # frameClone = image.copy()
          #cv2.imshow('your_face', frameClone)

                      image = cv2.resize(vs.read(), target_size)
                      batch_seg_map = sess.run('SemanticPredictions:0',
                                               feed_dict={'ImageTensor:0': [cv2.cvtColor(image, cv2.COLOR_BGR2RGB)]})
                      # visualization
                      seg_map = batch_seg_map[0]
                      seg_map[seg_map != 15] = 0

                      bg_copy = resized_background_image[img_num % num_files].copy()

                      # if fff == 0:
                      #     bg_copy = resized_background_image.copy()
                      # elif fff == 1:
                      #     bg_copy = resized_background_image2.copy()
                      # elif fff == 2:
                      #     bg_copy = resized_background_image3.copy()

                      mask = (seg_map == 15)
                      bg_copy[mask] = image[mask]

                      # create_colormap(seg_map).astype(np.uint8)
                      seg_image = np.stack(
                          (seg_map, seg_map, seg_map), axis=-1).astype(np.uint8)
                      gray = cv2.cvtColor(seg_image, cv2.COLOR_BGR2GRAY)

                      thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
                      major = cv2.__version__.split('.')[0]
                      if major == '3':
                          _, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                                                cv2.CHAIN_APPROX_SIMPLE)
                      else:
                          cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                                             cv2.CHAIN_APPROX_SIMPLE)

                      try:
                          cv2.drawContours(
                              bg_copy, cnts, -1, (randint(0, 255), randint(0, 255), randint(0, 255)), 2)
                      except:
                          pass

                      "angry", "disgust", "scared", "happy", "sad", "surprised",
                      "neutral"

                      ir = cv2.resize(bg_copy, (vs.real_width, vs.real_height))
                      """""

                      with torch.no_grad():

                              if a == "angry":
                                 ir1 = transfer.evaluate(style_model, ir, 1024, 'images/21styles/composition_vii.jpg', None, 1)

                              elif a=="disgust":
                                   ir1 = transfer.evaluate(style_model, ir, 1024, 'images/21styles/escher_sphere.jpg', None,1)

                              elif a =="scared":
                                   ir1 = transfer.evaluate(style_model, ir, 1024, 'images/21styles/feathers.jpg', None, 1)

                              elif a=="happy":
                                   ir1 = transfer.evaluate(style_model, ir, 1024, 'images/21styles/feathers.jpg', None, 1)

                              else:
                                  ir1 = transfer.evaluate(style_model, ir, 1024, 'images/21styles/candy.jpg', None, 1)
"""""
                      #    ir1 = transfer.evaluate(style_model, ir, 1024, 'images/21styles/candy.jpg', None, 0)
                      # print(ir1,'i am 22222')
                      # print(ir1.shape,'i am 22222')

                      # ir = cv2.imread('./output.jpg')
                      # print('i am 1',ir.shape)
                      # print('i am 1',ir)

                      # cv2.imshow("img", )

                      cv2.imshow('segmentation', np.uint8(ir))
                      if cv2.waitKey(1) & 0xFF == ord('q'):
                          break
                      elif cv2.waitKey(1) & 0xFF == ord('a'):
                          fff = 0
                      elif cv2.waitKey(1) & 0xFF == ord('b'):
                          fff = 1
                      elif cv2.waitKey(1) & 0xFF == ord('c'):
                          fff = 2
                      fps.update()

              # if fff == 0:
                    #     bg_copy = resized_background_image.copy()
                    # elif fff == 1:
                    #     bg_copy = resized_background_image2.copy()
                    # elif fff == 2:
                    #     bg_copy = resized_background_image3.copy()






                #print(ir1,'i am 22222')
               # print(ir1.shape,'i am 22222')

               # ir = cv2.imread('./output.jpg')
               # print('i am 1',ir.shape)
                #print('i am 1',ir)

                #cv2.imshow("img", )



                # out.write(ir)
    fps.stop()
    vs.stop()
    # out.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    graph = load_model1()
    segmentation(graph)
