import cv2
import math
import numpy as np

def distance(a, b):
    '''Pythagoras: finds the distance between two points (a1, a2), (b1, b2)'''
    deltax2 = (b[0]-a[0])**2
    deltay2 = (b[1]-a[1])**2
    ab = math.sqrt(deltax2 + deltay2)
    return ab

def dot_position(event, x, y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK: #double click
        a[0] = x
        a[1] = y
        print 'a is ' + str(a)

    if event == cv2.EVENT_RBUTTONDBLCLK: #double click
        b[0] = x
        b[1] = y
        print 'b is ' + str(b)

    return None



#input here
filepath = 'G:\\DT\\2018\\Jan\\Jan 12th'
videoname = 'distance_demo___1_12_11_42_13_0.avi'

#create two dots
a = [0,0]
b = [0,0]

#read the first frame of the video
video = cv2.VideoCapture(filepath+'\\'+videoname)
img = video.read()[1]
cv2.namedWindow('1st frame of '+videoname)
cv2.setMouseCallback('1st frame of '+videoname,dot_position)

#display the video
while(1):
    cv2.imshow('1st frame of '+videoname , img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pixel = distance(a,b)

ratio = 1/pixel

print 'each pixel correspond to ' + str(ratio) + 'mm'

# When everything done, release the capture
video.release()
cv2.destroyAllWindows()
