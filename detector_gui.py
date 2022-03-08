import os
from turtle import pendown
import cv2
import tkinter as tk
from tkinter import messagebox
from tkinter.filedialog import askopenfilename

from importlib.resources import path
import numpy as np
from glob import glob
import shutil

from PIL import ImageTk, Image
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


path = os.getcwd()
d = 'People'
files = os.path.join(path, d)
isdir = os.path.isdir(files)
if not isdir:
    os.mkdir(files)



class Application(tk.Frame):

    def __init__(self, master=None):
        super().__init__(master)

        self.pack()
        self.create_front()


    def create_front(self):

        def gui_destroy():
            root.destroy()

        self.uname_label = tk.Label(root, text="In-Video Person Detection System", font=('calibre',14,'bold'), bg='medium aquamarine')
        self.uname_label.place(x=110, y=40)

        self.go_upload = tk.Button(
            root, text="Upload Image of Person", font=('calibre',12,'bold'), bg='skyblue2', command=self.create_upload)
        self.go_upload.place(x=160, y=100)


        self.go_verify = tk.Button(
            root, text="Detect Person in Video", font=('calibre',12,'bold'), bg='gold2', command=self.create_verify)
        self.go_verify.place(x=170, y=160)


        quit = tk.Button(root, text="Exit", font=('calibre',12,'bold'), bg = "tomato", width=5, command=root.destroy)
        quit.place(x=220, y=220)


    def create_upload(self):

        def image_save(window, path0, path1):
            xx = self.name_entry.get()
            print("\n User name is : " + xx + "\n")

            sign_list = [path1]

            if path0=='':
                messagebox.showerror("Warning!", "Username can not be empty!")

            else:
                ff = 0
                for ee in sign_list:
                    file_exists = os.path.exists(ee)

                    if not file_exists:
                        messagebox.showerror("Warning!",
                                            "You must upload all 5 signatures!")
                        break
                    else:
                        ff += 1

                if ff==1:
                    cc = 0
                    for i in sign_list:
                        file = i
                        cc += 1
                        s_file = 'People/' + xx + '.jpg'
                        print("\n", file, "\n")
                        image = cv2.imread(file)

                        cv2.imwrite(s_file, image)

                    #     cv2.imshow(str(cc), image)
                    #     cv2.waitKey(0)

                    # cv2.destroyAllWindows()

                    dataset = tk.Label(root_u, text="Display Uploaded Image", font=('calibre',12,'bold'))
                    dataset.place(x=130, y=460)
                    
                    s_file1 = './People/' + path0 + '.jpg'
                    img1 = Image.open(s_file1)
                    img1 = img1.resize((80, 80), Image.ANTIALIAS)
                    img1 = ImageTk.PhotoImage(img1)
                    panel1 = tk.Label(root_u, image=img1)
                    panel1.place(x=20, y=500)

                    messagebox.showinfo("Success!",
                                            "Image Uploaded!!")
                    
                    dataset.destroy()

        def check_data():
            xx = self.name_entry.get()
            print("\n User name is : " + xx + "\n")

            file = 'People/' + xx + '.jpg'
            # print("\n", file, "\n")
            file_exists = os.path.exists(file)

            if not file_exists:
                messagebox.showwarning("Checked!",
                                    "User doesn't exist! Please Continue Uploading for Registration!")
            else:
                messagebox.showinfo("Checked!",
                                    "User exists! You can exit to Verify or Continue Upload and Update!")

        def browsefunc(ent):
            filename = askopenfilename(filetypes=([
                ("image", ".jpeg"),
                ("image", ".png"),
                ("image", ".jpg"),
            ]))
            ent.delete(0, tk.END)
            ent.insert(tk.END, filename)  # add this

        def gui_destroy():
            root_u.destroy()


        root_u = tk.Toplevel(self)
        root_u.title('Image Upload')
        root_u.geometry('500x350+650+50')

        self.uname_label = tk.Label(root_u, text="Upload Person Image", font=('calibre',14,'bold'), bg='skyblue2')
        self.uname_label.place(x=135, y=25)

        # creating a label for name using widget Label
        self.name_label = tk.Label(root_u, text = 'Person Name:', font=('calibre',10,'bold'))
        self.name_label.place(x=60, y=90)

        # creating a entry for input name using widget Entry
        self.name_entry = tk.Entry(root_u, bd=3, font=('calibre',10,'normal'))
        self.name_entry.place(x=170, y=90)

        # creating a button using the widget button that will call the submit function
        sub_btn = tk.Button(root_u, text = 'Check Data', font=('calibre',10,'normal'), command = check_data)
        sub_btn.place(x=350, y=88)


        # Image 1
        self.img_message = tk.Label(root_u, text="Image:", font=('calibre',10,'bold'))
        self.img_message.place(x=60, y=140)
        # Image Submit
        self.image_path_entry1 = tk.Entry(root_u, bd=3, font=('calibre',10,'normal'))
        self.image_path_entry1.place(x=170, y=140)
        # Browse Button
        self.img_browse_button = tk.Button(
            root_u, text="Browse", font=('calibre',10,'normal'), command=lambda: browsefunc(ent=self.image_path_entry1))
        self.img_browse_button.place(x=350, y=138)


        # registered Button
        self.register_button = tk.Button(
            root_u, text="Register", font=('calibre',12,'bold'), bg='gold2', command=lambda: image_save(window=root_u,
                                                                        path0=self.name_entry.get(),
                                                                        path1=self.image_path_entry1.get(),), width=8)
        self.register_button.place(x=198, y=200)


        # Exit Button
        go_exit = tk.Button(
            root_u, text="Exit", font=('calibre',12,'bold'), bg='tomato', command=lambda: gui_destroy(), width=5)
        go_exit.place(x=214, y=250)

        root_u.mainloop()


    def create_verify(self):
        # Mach Threshold
        THRESHOLD = 50

        root_v=tk.Toplevel(self)
        root_v.title("Person Detection")

        # setting the windows size
        root_v.geometry("500x500+650+50")


        # defining a function that will get the name and password and print them on the screen
        def view_data():
            name = self.name_entry.get()
            
            print("\n The name is : " + name + "\n")

            for i in range(1):
                file = 'People/' + name + '.jpg'
                print("\n", file, "\n")
                image = cv2.imread(file)

                image = cv2.resize(image, (300, 300))

                cv2.imshow(str(i+1), image)
                cv2.waitKey(0)

            cv2.destroyAllWindows()


        def check_data():
            name = self.name_entry.get()
            print("\n Person name is : " + name + "\n")

            file = 'People/' + name + '.jpg'
            # print("\n", file, "\n")
            file_exists = os.path.exists(file)

            if not file_exists:
                messagebox.showerror("Warning!",
                                    "User doesn't exist! Please Enter Correct Username!")
            else:
                messagebox.showinfo("Checked!",
                                    "User exists! Please Continue Upload to Verify!")


        def browsefunc(ent):
            filename = askopenfilename(filetypes=([
                ("video", ".mp4"),
                ("video", ".avi"),
                ("video", ".mkv"),
            ]))
            ent.delete(0, tk.END)
            ent.insert(tk.END, filename)  # add this


        def checkSimilarity(window, path0, path1):

            pending = tk.Label(root_v, text="Video Processing ...", font=('calibre',12,'bold'))
            pending.place(x=140, y=300)

            if path0=='' or path1=='':
                messagebox.showerror("Warning!", "Username or Uploaded Image can not be empty while varifying!")

            else:
                ch_file = './People/' + path0 + '.jpg'
                file_exists = os.path.exists(ch_file)

                if not file_exists:
                    messagebox.showerror("Warning!", "User does not exist in Database! Please enter Username correctly for verifying! Or, Exit and Go to User Registration")

                else:

                    def main(_argv):

                        yy = strftime("%d-%b-%Y_%H-%M", gmtime())
                        # print(yy)

                        # Source Images
                        images = []
                        classNames = []


                        path = 'People'
                        name = path0
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


                        # For only video
                        FLAGS.video = path1
                        FLAGS.output = "./detections/video_output.mp4"

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
                                                cv2.rectangle(frame_i,(x1,y1),(x2,y2),(0,255,0),1)
                                                cv2.rectangle(frame_i,(x1-30,y2+20),(x2+30,y2),(0,255,0),cv2.FILLED)
                                                cv2.putText(frame_i, name, (x1-28,y2+18), cv2.FONT_HERSHEY_COMPLEX,0.7,(255,255,255),2)
        
                                                frame_i = cv2.cvtColor(frame_i, cv2.COLOR_BGR2RGB)
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
                                        # y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
                                        cv2.rectangle(frame,(x1-30,y2+20),(x2+30,y2),(0,255,0),cv2.FILLED)
                                        cv2.putText(frame, name, (x1-28,y2+18), cv2.FONT_HERSHEY_COMPLEX,0.7,(255,255,255),2)

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


                        # dataset = tk.Label(root_v, text="Display Images in Database", font=('calibre',12,'bold'))
                        # dataset.place(x=140, y=300)
                        
                        # s_file1 = './People/' + path0 + '.jpg'
                        # img1 = Image.open(s_file1)
                        # img1 = img1.resize((80, 80), Image.ANTIALIAS)
                        # img1 = ImageTk.PhotoImage(img1)
                        # panel1 = tk.Label(root_v, image=img1)
                        # panel1.place(x=20, y=340)

                        # d_result = tk.Label(root_v, text="Detection Result", font=('calibre',13,'bold'))
                        # d_result.place(x=20, y=450)


                        d_result = tk.Label(root_v, text="Detection Result", font=('calibre',13,'bold'))
                        d_result.place(x=20, y=370)

                        if person_count != 0:
                            print('\n\n--- Result: Positive! ', name, 'is present in the video. ---\n')

                            got_frames = "./detections/person_" + yy + "/"

                            print('The captured frames are:')
                            for f in os.listdir(got_frames):
                                fr_name = f.split('.')[0]
                                print(fr_name)
                            print()

                            valid_result = tk.Label(root_v, text="Result: Positive! " + name + " is present in the video!!", font=('calibre',11,'bold'), bg='green')
                            valid_result.place(x=20, y=420)
                            messagebox.showinfo("Success: Person Detected!",
                                                "Target person is present in the video!!")
                            valid_result.destroy()

                        else:
                            print('\n\n--- Result: Negative! ', name, 'is not present in the video. ---\n\n')

                            fail_result = tk.Label(root_v, text="Result: Negative! " + name + " is not present in the video!!", font=('calibre',11,'bold'), bg='red')
                            fail_result.place(x=20, y=420)
                            messagebox.showerror("Failure: Person Not Detected.",
                                                "Target person is not present in the video!!")
                            fail_result.destroy()

                        cv2.destroyAllWindows()

                        pending.destroy()
                        d_result.destroy()


                    if __name__ == '__main__':
                        try:
                            app.run(main)
                        except SystemExit:
                            pass


            return True


        def gui_destroy():
            root_v.destroy()



        self.uname_label = tk.Label(root_v, text="In-video Person Detection", font=('calibre',14,'bold'), bg='medium aquamarine')
        self.uname_label.place(x=140, y=25)

        # creating a label for name using widget Label
        self.name_label = tk.Label(root_v, text = 'Username:', font=('calibre',10,'bold'))
        self.name_label.place(x=60, y=90)

        # creating an entry for input name using widget Entry
        self.name_entry = tk.Entry(root_v, bd=3, font=('calibre',10,'normal'))
        self.name_entry.place(x=170, y=90)

        # creating a button using the widget button that will check the available data
        self.sub_btn = tk.Button(root_v, text = 'Check Data', font=('calibre',10,'normal'), command = check_data)
        self.sub_btn.place(x=340, y=85)


        # Upload
        self.img_message = tk.Label(root_v, text="Input Video:", font=('calibre',10,'bold'))
        self.img_message.place(x=60, y=140)
        # Image Submit
        self.image_path_entry1 = tk.Entry(root_v, bd=3, font=('calibre',10,'normal'))
        self.image_path_entry1.place(x=170, y=140)
        # Browse Button
        self.img_browse_button = tk.Button(
            root_v, text="Browse", font=('calibre',10,'normal'), command=lambda: browsefunc(ent=self.image_path_entry1))
        self.img_browse_button.place(x=340, y=135)



        # Verify Button
        self.verify_button = tk.Button(
            root_v, text="Verify", font=('calibre',12,'bold'), bg='gold2', command=lambda: checkSimilarity(window=root_v, path0=self.name_entry.get(), path1=self.image_path_entry1.get(),), width=8)
        self.verify_button.place(x=198, y=190)


        # Exit Button
        self.go_exit = tk.Button(
            root_v, text="Exit", bg='tomato', font=('calibre',12,'bold'), command=lambda: gui_destroy(), width=5)
        self.go_exit.place(x=215, y=240)


        # performing an infinite loop for the window to display
        root_v.mainloop()



root = tk.Tk()
root.configure(bg='wheat1')
root.geometry("500x300+50+50")

app_g = Application(master=root)
app_g.master.title("Person Detection & Identification System")
app_g.mainloop()
