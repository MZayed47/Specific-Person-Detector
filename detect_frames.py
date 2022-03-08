import os
import re
import cv2
from glob import glob
import shutil

yy = '03-Mar-2022_04-26'

for file in glob("./detections/crop_" + yy + "/*/", recursive = True):
    # Get the file names
    ff = os.path.normpath(file)
    xx = os.path.basename(ff)

    # Get all the images in the folders
    for i in os.listdir(file):
        # print(i)
        ii = './detections/crop_' + yy + '/' + xx + '/' + i
        image = cv2.imread(ii)
        cv2.imwrite('./detections/crop_' + yy + '/' + xx + '_' + i, image)
    
    # os.remove(file)
    shutil.rmtree(file)
        # try:
        #     cv2.imshow('Images',image)
        # except:
        #     pass
        # cv2.waitKey(0)
    
    # cv2.destroyAllWindows()

