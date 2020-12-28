import numpy as np
import tensorflow as tf2
import transfer
"""""
v_tf = tf.__version__.split('.')[0]
if v_tf == '2':
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
# import imutils
"""
import torch
from net import Net
import tensorflow.compat.v1 as tf
import functools
import os

from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow_hub as hub
from tensorflow.python.keras.backend import set_session

import os
from stuff.helper import FPS2, WebcamVideoStream
# from skimage import measure
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

EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised",
            "neutral"]

img_num = 0
mike_flag = True



def crop_center(image):
  """Returns a cropped square image."""
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image

@functools.lru_cache(maxsize=None)
def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True,):
  """Loads and preprocesses images."""
  # Cache image file locally.
  #image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  #img = plt.imread(image_path).astype(np.float32)[np.newaxis, ...]
  img = plt.imread(image_url).astype(np.float32)[np.newaxis, ...]


  if img.max() > 1.0:
    img = img / 255.
  if len(img.shape) == 3:
    img = tf2.stack([img, img, img], axis=-1)
  img = crop_center(img)
  img = tf2.image.resize(img, image_size, preserve_aspect_ratio=True)
  print(img,'fffff')
  print(type(img),'ggggg')
  return img


def change_image(img, image_size=(256, 256), preserve_aspect_ratio=True,):
  """Loads and preprocesses images."""
  # Cache image file locally.
  #image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  #img = plt.imread(image_path).astype(np.float32)[np.newaxis, ...]

  if img.max() > 1.0:
    img = img / 255.
  if len(img.shape) == 3:
    img = tf2.stack([img, img, img], axis=-1)
  img = crop_center(img)
  img = tf2.image.resize(img, image_size, preserve_aspect_ratio=True)
  print(img,'fffff')
  print(type(img),'ggggg')
  return img

def show_n(images, titles=('',)):
  n = len(images)
  image_sizes = [image.shape[1] for image in images]
  w = (image_sizes[0] * 6) // 320
  plt.figure(figsize=(w  * n, w))
  gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)
  for i in range(n):
    plt.subplot(gs[i])
    plt.imshow(images[i][0], aspect='equal')
    plt.axis('off')
    plt.title(titles[i] if len(titles) > i else '')
  plt.show()




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

    return graph, emotion_classifier


def predict(faces, gray, graph, model):
    """
    图片文字方向预测
    """
    # emotion_classifier = load_model(emotion_model_path, compile=False)
    # graph = tf.get_default_graph()
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

    # cv2.namedWindow('segmentation', 16)  # 16 means WINDOW_GUI_NORMAL, to disable right click context menu

    # cv2.setMouseCallback('segmentation', next_bg)
    # graph1,model1=loadrecongmodel(emotion_model_path)

    sess = tf.Session()
    graph = tf.get_default_graph()

    # 在model加载前添加set_session
    set_session(sess)
    emotion_classifier = load_model(emotion_model_path, compile=False)

    graph_face_detection = tf.Graph()
    sess_face_detection = tf.Session(graph=graph)

    sess_face_recognition = tf.Session(graph=detection_graph)

    # sess_face_detection.run(operations)

    # sess_face_recognition.run(operations)

    img_num = 0
    modules = hub.load('./models/magenta_arbitrary-image-stylization-v1-256_2.tar')

    while vs.isActive():
        image = imutils.resize(vs.read(), width=300)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
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

            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]
            print(label)

        # a = predict(faces, gray,graph1,model1)
        # print(a)
        # canvas = np.zeros((250, 300, 3), dtype="uint8")
        # frameClone = image.copy()
        # cv2.imshow('your_face', frameClone)

        image = cv2.resize(vs.read(), target_size)
        #  batch_seg_map = sess.run('SemanticPredictions:0',
        #          feed_dict={'ImageTensor:0': [cv2.cvtColor(image, cv2.COLOR_BGR2RGB)]})
        # visualization
        batch_seg_map = sess_face_recognition.run('SemanticPredictions:0',
                                                  feed_dict={'ImageTensor:0': [cv2.cvtColor(image, cv2.COLOR_BGR2RGB)]})

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
        a = load_image('./images/00.jpg')
        #print(a, 'iam    s')

        ir = cv2.resize(bg_copy, (vs.real_width, vs.real_height))
        ir = ir[np.newaxis, ...]

        b = change_image(ir, (640, 480))
        #print(b, 'i am go bbbbb')
        aaa = modules(b, a)
        #print(aaa, 'DDFSF')
        # ir1 = transfer.evaluate(style_model, ir, 1024, 'images/21styles/candy.jpg', None, 0)
        # print(ir1,'i am 22222')
        # print(ir1.shape,'i am 22222')

        # ir = cv2.imread('./output.jpg')
        # print('i am 1',ir.shape)
        # print('i am 1',ir)

        # cv2.imshow("img", )
        #print(aaa[0],'i am aaaa')
        ir=np.squeeze(aaa[0].numpy(), axis=0)
        ir = cv2.cvtColor(ir, cv2.COLOR_RGB2BGR)
        #print(ir.shape)
        #show_n([aaa[0], a], ['Content image', 'Style image'])

        ir = cv2.flip(ir, 1)
        cv2.imshow('segmentation', ir)
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

    # print(ir1,'i am 22222')
    # print(ir1.shape,'i am 22222')

    # ir = cv2.imread('./output.jpg')
    # print('i am 1',ir.shape)
    # print('i am 1',ir)

    # cv2.imshow("img", )

    # out.write(ir)
    fps.stop()
    vs.stop()
    # out.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    graph = load_model1()
    segmentation(graph)
