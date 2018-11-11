from geometry_helpers import *
import cv2
import numpy as np
import os

enter_key = 13
escape_key = 27
monitor_size = (2560, 1440)

''' DT testing code
folder = 'C:\\DT files\\Julie Semmelhack Lab\\possible head fixed strike\\test'
filenames = os.listdir(folder)
avis = [filename for filename in filenames if os.path.splitext(filename)[1] == '.avi']
filepath = os.path.join(folder, avis[0])

image = 1
thresh = 200
cap = cv2.VideoCapture(filepath)
image = cap.read()[1]
c_1, th, l_c, l_th, r_c, r_th = frameData(image, thresh)  # tag
print c_1
'''


def cropImage(image, ROI):
    x1, y1 = ROI[0]
    x2, y2 = ROI[1]
    cropped = image[y1:y2+1, x1:x2+1]
    return cropped


def applyThreshold(image, value, threshold='to_zero'):
    if threshold == 'to_zero':
        ret, new = cv2.threshold(image, value, 255, cv2.THRESH_TOZERO)
    elif threshold == 'otsu':
        ret, new = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif threshold == 'binary':
        ret, new = cv2.threshold(image, value, 255, cv2.THRESH_BINARY)
    else:
        new = image
    new = new.astype('uint8')
    return new


def subtractBackground(image, background):
    bg = background.astype('i4')
    new = bg - image
    new = np.clip(new, 0, 255)
    new = new.astype('uint8')
    return new


def findContours(image, offset=None):
    new = np.copy(image)
    #new = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # DT added this function to convert images into single channel
    contours, hierarchy = cv2.findContours(new, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda contour: cv2.contourArea(contour))
    contours.reverse()
    return contours


def drawContours(image, contours, c=0, t=1, thresh = 200,  c_1 = None, l_c= None, r_c=None):
    new = np.copy(image)
    color = (0, 255, 0)
    if l_c != None:
       mid_point = findMidpoint(l_c,r_c)
       p1 = (int(mid_point[0]),int(mid_point[1]))
       p2 = (int(c_1[0]),int(c_1[1]))
       cv2.circle(new, p1, 3, color, -1)
       cv2.circle(new, p2, 3, color, -1)
    cv2.drawContours(new, contours, -1, c, t)

    return new


def equaliseHist(image):
    equalised = cv2.equalizeHist(image)
    return equalised


class Video(object): #Q why inherit object?

    def __init__(self, filepath, background=False):

        self.name = filepath
        self.object = cv2.VideoCapture(filepath)
        # self.framerate = self.object.get(cv2.cv.CV_CAP_PROP_FPS)
        self.framecount = int(self.object.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))     #Number of frames in the video file
        self.shape = (self.object.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH), self.object.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

        self.framenumber = 0
        self.limit_frames = [0, self.framecount]  #Number of frames in the video file

        if background:
            self.background = self.intensityProjection()
        else:
            self.background = None

        self.displays = []

    ############################

    def grabFrame(self):
        self.object.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, self.framenumber)  # 0-based index of the frame to be decoded/captured next
        ret, frame = self.object.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = np.asarray(frame)     # Convert the input to an array
        return frame

    def updateFramenumber(self, n):
        if self.limit_frames[0] <= n <= self.limit_frames[1]:
            self.framenumber = n
        elif n < self.limit_frames[0]:
            self.framenumber = self.limit_frames[0]
        else:
            self.framenumber = self.limit_frames[1]

    def grabFrameN(self, n):
        self.updateFramenumber(n)
        frame = self.grabFrame()
        return frame

    ############################

    def intensityProjection(self):
        print 'calculating background...',
        background = self.grabFrameN(0)
        for i in range(1, self.framecount):
            img = self.grabFrameN(i)
            brighter = np.transpose(np.where(img >= background))
            for j, k in brighter:
                background[j, k] = img[j, k]
        self.updateFramenumber(0)
        print 'complete!'
        return background


    def importBackground(self, tiff_filepath):
        background = cv2.imread(tiff_filepath, 0)
        self.background = background

    ############################

    def addDisplay(self, winname, displayType = 'normal', framebar=True, displayFunction=None, displayKwargs=None): #tag
        if displayType == 'normal':
            display = Display(self, winname, displayFunction, displayKwargs)   #tag
        elif displayType == 'selection':
            display = SelectorDisplay(self, winname, displayFunction, displayKwargs)
        elif displayType == 'event':
            display = EventDisplay(self, winname, displayFunction, displayKwargs)
        else:
            if type(displayType) != str:
                raise TypeError('displayType is not a string!')
            else:
                raise ValueError('invalid displayType')

        self.displays.append(display)

        if framebar:
            self.addFramebar(winname)
        # display.updateDisplay() ### HERE OR IN DISPLAY.CREATE_DISPLAY()

        return display

    def removeDisplay(self, winname):
        display = self.getDisplay(winname)
        display.destroyDisplay()
        self.displays.remove(display)

    def getDisplay(self, winname):
        names = [display.window for display in self.displays]  #tag
        index = names.index(winname)
        return self.displays[index]

    def updateDisplays(self):
        img = self.grabFrame()
        for display in self.displays:
            display.image = img
            display.updateDisplay()

    def addFramebar(self, winname):
        display = self.getDisplay(winname)
        start = self.framenumber-self.limit_frames[0]
        max_val = self.limit_frames[1]-self.limit_frames[0]
        cv2.createTrackbar('frame', winname, start, max_val, self.framebarChange)    #framebarChange can be use this way? it's arguement required function!
        display.trackbars['frame'] = self.framenumber

    def framebarChange(self, value):         # the value here should correspond to the start in addFramebar
        frame = self.limit_frames[0]+value
        self.updateFramenumber(frame)
        for display in self.displays:
            if 'frame' in display.trackbars.keys():
                if 'start' in display.trackbars.keys():
                    min_val = display.trackbars['start']
                    if frame < min_val:
                        self.updateFramenumber(min_val)
                if 'end' in display.trackbars.keys():
                    max_val = display.trackbars['end']
                    if frame > max_val:
                        self.updateFramenumber(max_val)
                cv2.setTrackbarPos('frame', display.window, self.framenumber-self.limit_frames[0])  #Q I need to excatly how frame number is calculated?
                display.trackbars['frame'] = frame
        self.updateDisplays()

    def addThreshbar(self, winname, thresh_name, initial):
        display = self.getDisplay(winname)
        cv2.createTrackbar(thresh_name, winname, initial, 255, self.threshbarChange)
        try:
            display.trackbars['thresholds'][thresh_name] = initial
        except KeyError:                             #Python raises a KeyError whenever a dict() object is requested (using the format a = adict[key]) and the key is not in the dictionary.
            display.trackbars['thresholds'] = {thresh_name: initial}

    def threshbarChange(self, dummy):
        for display in self.displays:
            if 'thresholds' in display.trackbars.keys():
                for thresh_name in display.trackbars['thresholds'].keys():         #Q trackbars as dictionary???
                    threshval = cv2.getTrackbarPos(thresh_name, display.window)    #The function returns the current position of the specified trackbar.
                    display.trackbars['thresholds'][thresh_name] = threshval
        self.updateDisplays()

    def updateLimits(self, lower, upper):
        if 0 <= lower <= upper:
            self.limit_frames[0] = int(lower)
        else:
            self.limit_frames[0] = 0
        if lower <= upper <= self.framecount:
            self.limit_frames[1] = int(upper)
        else:
            self.limit_frames[1] = self.framecount
        self.updateFramenumber(self.framenumber)


####################################################################################


class Display(object):

    def __init__(self, video, winname, displayFunction, displayKwargs):

        self.video = video
        self.window = winname
        self.trackbars = {}

        self.image = self.video.grabFrame()
        self.displayFunction = displayFunction
        self.displayKwargs = displayKwargs

        self.createDisplay() #tag

    ############################

    def createDisplay(self):
        cv2.namedWindow(self.window)
        self.updateDisplay() ### HERE OR IN VIDEO.ADD_DISPLAY()

    def destroyDisplay(self):
        cv2.destroyWindow(self.window)

    def updateDisplay(self):
        image = self.image
        image = cv2.resize(image, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)

        if self.displayFunction:
            if self.displayKwargs:
                image = self.displayFunction(image, **self.displayKwargs)
            else:
                image = self.displayFunction(image)
        cv2.imshow(self.window, image)               ###a window have been created and showed

    ############################


class SelectorDisplay(Display):

    def __init__(self, video, winname, displayFunction, displayKwargs):

        Display.__init__(self, video, winname, displayFunction, displayKwargs)

        self.selection = False
        self.p1 = None
        self.p2 = None

        cv2.setMouseCallback(self.window, self.updateClick)

    ############################
    def updateDisplay(self):
        self.image = self.video.grabFrame()
        image = self.image
        #image = cv2.resize(image, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)

        if self.displayFunction:
            if self.displayKwargs:
                image = self.displayFunction(image, **self.displayKwargs)
            else:
                image = self.displayFunction(image)
        try:
            if self.p1 and self.p2:
                cv2.rectangle(image, self.p1, self.p2, 0)
        except AttributeError:
            pass
        cv2.imshow(self.window, image)

    ############################

    def updateClick(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.selection = False
            self.p1 = (x, y)
            self.p2 = None
        elif event == cv2.EVENT_LBUTTONUP:
            self.p2 = (x, y)
            self.selection = True
        elif event == cv2.EVENT_RBUTTONUP:
            self.selection = False
            self.p1 = None
            self.p2 = None
        elif not self.selection:
            self.p2 = (x, y)

        self.updateDisplay()


class EventDisplay(Display):

    def __init__(self, video, winname, displayFunction, displayKwargs):

        Display.__init__(self, video, winname, displayFunction, displayKwargs)

        max_val = self.video.limit_frames[1] - self.video.limit_frames[0]

        cv2.createTrackbar('start', winname, 0, max_val, self.trackbarChange)
        cv2.createTrackbar('end', winname, 0, max_val, self.trackbarChange)

        self.trackbars['start'] = self.video.limit_frames[0]
        self.trackbars['end'] = self.video.limit_frames[1]

    ############################

    def trackbarChange(self, value):
        start_val = cv2.getTrackbarPos('start', self.window)
        end_val = cv2.getTrackbarPos('end', self.window)
        self.trackbars['start'] = self.video.limit_frames[0] + start_val
        self.trackbars['end'] = self.video.limit_frames[0] + end_val
        if end_val < start_val:
            cv2.setTrackbarPos('end', self.window, start_val)
            self.trackbars['end'] = self.trackbars['start']
        self.video.framebarChange(value)
        self.updateDisplay()


####################################################################################

def scrollVideo(video):
    video.addDisplay(video.name)
    cv2.waitKey(0)
    video.removeDisplay(video.name)


def selectROI(video, name=''):
    winname = 'select ROI {}'.format(name)
    video.addDisplay(winname, displayType='selection') #tag
    k = cv2.waitKey(0)
    display = video.getDisplay(winname)
    if k == enter_key and display.selection:
        points = [display.p1, display.p2]
        x_min = min([p[0] for p in points])
        y_min = min([p[1] for p in points])
        x_max = max([p[0] for p in points])
        y_max = max([p[1] for p in points])
        roi_coords = (x_min, y_min), (x_max, y_max)
        video.removeDisplay(winname)
        return roi_coords
    else:
        print 'WARNING: no ROI selected!'
        video.removeDisplay(winname)
        return

def displayCropImage(image, **kwargs):
    if kwargs['roi'] is not None:
        img = cropImage(image, kwargs['roi'])
    else:
        img = image
    return img

def selectmouthROI(video, name='', roi=None):
    winname = 'select mouth ROI {}'.format(name)
    displayKwargs = dict(video=video, winname=winname, roi=roi)
    video.addDisplay(winname, displayType='selection', displayFunction=displayCropImage, displayKwargs=displayKwargs ) #DT-tag
    # image = self.displayFunction(image, **self.displayKwargs)

    k = cv2.waitKey(0)
    display = video.getDisplay(winname)
    if k == enter_key and display.selection:
        points = [display.p1, display.p2]
        x_min = min([p[0] for p in points])
        y_min = min([p[1] for p in points])
        x_max = max([p[0] for p in points])
        y_max = max([p[1] for p in points])
        roi_coords = (x_min, y_min), (x_max, y_max)
        video.removeDisplay(winname)
        return roi_coords
    else:
        print 'WARNING: no ROI selected!'
        video.removeDisplay(winname)
        return

def selectEvent(video):
    winname = video.name
    video.addDisplay(winname, displayType='event')
    k = cv2.waitKey(0)
    display = video.getDisplay(winname)
    if k == enter_key:
        start_frame = display.trackbars['start']
        end_frame = display.trackbars['end']
        video.removeDisplay(winname)
        return (start_frame, end_frame)
    else:
        # print 'no event selected'
        video.removeDisplay(winname)
        return
