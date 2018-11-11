from filepicker import * #SQ*
from eye_tracker import *
from eye_tracker_helpers import *
import pandas as pd
import os
from matplotlib import pyplot as plt
import csv
import copy


def nothing():
    return 'hello'

if __name__ == "__main__":

    ### USER INPUT HERE ###
    ### SELECT A FOLDER TO ANALYZE, FIND ALL AVI FILES IN THAT FOLDER ###
    #folder = pickdir()
    folder = 'G:\\DT\\2018\\Jan\\Jan 12th\\results\\mouth analysis'   # this should be the folder where you put all the videos for analysis
    csv_filename = ''   # this should be the file where you stored the striking frame
    filenames = os.listdir(folder)
    avis = [filename for filename in filenames if os.path.splitext(filename)[1] == '.avi']


    for avi in avis:
        print avi  #

        video = Video(os.path.join(folder, avi))
        ROI = selectROI(video)

        for frame in range(video.framecount):

            img = video.grabFrameN(frame)

            if ROI is not None:
                img = cropImage(img, ROI)

            ret, thresh1 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

            cv2.namedWindow(avi)
            cv2.createTrackbar('Frame',avi,0,video.framecount,nothing)
            cv2.imshow(avi,thresh1)
            cv2.waitKey(0)





