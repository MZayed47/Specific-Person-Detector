from importlib.resources import path
import os
import cv2
import numpy as np
from glob import glob
import shutil

from PIL import ImageGrab
from PIL import Image

import time
from time import gmtime, strftime
from datetime import datetime
import tensorflow as tf

# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


physical_devices = tf.config.experimental.list_physical_devices('GPU')

if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


import face_recognition
import csv


flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/video.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.70, 'iou threshold')
flags.DEFINE_float('score', 0.70, 'score threshold')
flags.DEFINE_boolean('count', False, 'count objects within video')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'print info on detections')
flags.DEFINE_boolean('crop', False, 'crop detections from images')
flags.DEFINE_boolean('plate', False, 'perform license plate recognition')
flags.DEFINE_boolean('person', False, 'perform person detection')
flags.DEFINE_boolean('frames', False, 'get the frames with persons')
flags.DEFINE_boolean('identify', False, 'Identify the target person')


# while True:
#     print("\n\nEnter 0 for WEBCAM, or 1 for Existing Video: ")
#     nn = input()
#     if nn == '0':
#         nn=0
#         print('\nStarting WEBCAM ... \n\n')
#         break
#     elif nn == '1':
#         nn=1
#         print('\nLoading Video ... \n\n')
#         break
#     else:
#         print('\nSorry! Enter Correct Option.\n')


def main(_argv):

    yy = strftime("%d-%b-%Y_%H-%M", gmtime())
    # print(yy)

    # Source Images
    images = []
    classNames = []

    # path = 'People'
    # myList = os.listdir(path)
    # print(myList)

    # Image Names
    # for cl in myList:
    #     if cl == "Mashrukh Zayed" + ".jpg":
    #         curImg = cv2.imread(f'{path}/{cl}')
    #         images.append(curImg)
    #         classNames.append(os.path.splitext(cl)[0])
    # # print(classNames)

    path = 'People'
    name = "Tadib Chowdhury"
    cl = name + ".jpg"
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

    # Face Encodings
    def findEncodings(images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList

    encodeListKnown = findEncodings(images)
    # print('Encoding Complete')
    # print("Number of Records: ",len(encodeListKnown))


    ###############################################################

    FLAGS.weights = './checkpoints/yolov4-416'
    FLAGS.model = 'yolov4'
    FLAGS.size = 416

    # # For a recorded video
    # if nn==1:
    #     FLAGS.video = './data/video/room.mp4'
    #     FLAGS.output = "./detections/room_output.mp4"

    # # # For Live Webcam
    # if nn==0:
    #     FLAGS.video = '0'
    #     FLAGS.output = "./detections/webcam_video.mp4"

    # For only video
    FLAGS.video = './data/video/today.mp4'
    FLAGS.output = "./detections/today_output.mp4"

    # FLAGS.person = True
    FLAGS.crop = True
    FLAGS.frames = True
    FLAGS.identify = True

    person_count = 0

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video
    # get video name by using split method
    video_name = video_path.split('/')[-1]
    video_name = video_name.split('.')[0]


    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None


    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        vid_fps = fps
        # print(vid_fps)
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    ##############################################################################
    # Main Loop Starts
    frame_num = -1

    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_num += 1
            image = Image.fromarray(frame)
        else:
            print('\n--- Video has ended or failed. Check the output or try a different video format! ---')
            break
    
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
 
        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to allow detections for only people)
        allowed_classes = ['person']


        # if crop flag is enabled, crop each detection and save it as new image
        if FLAGS.crop:
            crop_rate = int(vid_fps/2) # capture images every so many frames (ex. crop photos every 150 frames)
            crop_path = os.path.join(os.getcwd(), 'detections', 'crop_' + yy)
            try:
                os.mkdir(crop_path)
            except FileExistsError:
                pass
            if frame_num % crop_rate == 0:
                final_path = os.path.join(crop_path, 'frame_' + str(frame_num))
                try:
                    os.mkdir(final_path)
                except FileExistsError:
                    pass          
                crop_objects(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), pred_bbox, final_path, allowed_classes)
            else:
                pass


        if FLAGS.frames:
            for file in glob("./detections/crop_" + yy + "/*/", recursive = True):
                # Get the file names
                ff = os.path.normpath(file)
                xx = os.path.basename(ff)

                # Get all the images in the folders
                for i in os.listdir(file):
                    # print(i)
                    ii = './detections/crop_' + yy + '/' + xx + '/' + i
                    image_i = cv2.imread(ii)
                    cv2.imwrite('./detections/crop_' + yy + '/' + xx + '_' + i, image_i)
                shutil.rmtree(file)


        if FLAGS.identify:
            file = "./detections/crop_" + yy + "/"

            for i in os.listdir(file):
                img_name = i.split('.')[0]
                # print(img_name)
                try:
                    frame_i = cv2.imread("./detections/crop_" + yy + "/" + i)
                    frame_i = cv2.cvtColor(frame_i, cv2.COLOR_BGR2RGB)
                    facesCurFrame = face_recognition.face_locations(frame_i)
                    encodesCurFrame = face_recognition.face_encodings(frame_i, facesCurFrame)

                    person_path = os.path.join(os.getcwd(), 'detections', 'person_' + yy)
                    isdir = os.path.isdir(person_path)
                    if not isdir:
                        os.mkdir(person_path)

                    for encodeFace,faceLoc in zip(encodesCurFrame, facesCurFrame):
                        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                        # print(faceDis)
                        matchIndex = np.argmin(faceDis)

                        if matches[matchIndex]:
                            name = classNames[matchIndex].upper()
                            # print(name, 'is present in the video.\n')

                            y1,x2,y2,x1 = faceLoc
                            # y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                            cv2.rectangle(frame_i,(x1,y1),(x2,y2),(0,255,0),2)
                            cv2.rectangle(frame_i,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                            cv2.putText(frame_i, name, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX,0.7,(255,255,255),2)

                            cv2.imwrite(person_path + '/' + img_name + '_' + name + '.jpg', frame_i)

                            person_count += 1

                except:
                    pass


        if FLAGS.count:
            # count objects found
            counted_classes = count_objects(pred_bbox, by_class = True, allowed_classes=allowed_classes)
            # loop through dict and print
            for key, value in counted_classes.items():
                pass
                # print("Number of {}s: {}".format(key, value))
            image = utils.draw_bbox(frame, pred_bbox, FLAGS.info, counted_classes, allowed_classes=allowed_classes, read_plate=FLAGS.plate)
        else:
            image = utils.draw_bbox(frame, pred_bbox, FLAGS.info, allowed_classes=allowed_classes, read_plate=FLAGS.plate)


        if FLAGS.person:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            facesCurFrame = face_recognition.face_locations(frame)
            encodesCurFrame = face_recognition.face_encodings(frame,facesCurFrame)

            person_path = os.path.join(os.getcwd(), 'detections', 'person_' + yy)
            isdir = os.path.isdir(person_path)
            if not isdir:
                os.mkdir(person_path)

            for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
                # print(faceDis)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()
                    print(name,'is present in the video.\n')

                    final_path = os.path.join(person_path, 'frame_' + str(frame_num) + '_' + name)

                    y1,x2,y2,x1 = faceLoc
                    y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                    cv2.putText(frame, name, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX,0.7,(255,255,255),2)

                    cv2.imwrite(final_path + '.jpg', frame)


        fps = 1.0 / (time.time() - start_time)
        # print("FPS: %.2f" % fps)
        result = np.asarray(image)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("result", result)
        
        if FLAGS.output:
            out.write(result)

        if cv2.waitKey(1) & 0xFF == ord('q'): break


    if person_count != 0:
        print('\n\n--- Result: Positive! ', name, 'is present in the video. ---\n')
        got_frames = "./detections/person_" + yy + "/"

        print('The captured frames are:')
        for f in os.listdir(got_frames):
            fr_name = f.split('.')[0]
            print(fr_name)
        print()
    else:
        print('\n\n--- Result: Negative! ', name, 'is not present in the video. ---\n\n')


    cv2.destroyAllWindows()



if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
