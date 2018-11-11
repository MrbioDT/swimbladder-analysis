# -*- coding: UTF-8 -*-

from video_handling import *
from geometry_helpers import *
from matplotlib import pyplot as plt



# CONTOUR FUNCTIONS #

dt = 1

def dt_peak_detection(list,threshold):
    length = len(list)
    threshold = threshold   #this is the threshold for abnormal peak detection
    for i in range(length):
        if i != 0:
            old = list[i-1]
            current = list[i]
            if abs(old-current) >= threshold:
                list[i] = old
    return list

def dt_plot_contour(contour):
    # dt_function: the funciton can plot the controur, arguement is a numpy.ndarray kind of coutour
    for i in range(contour.shape[0]):
        # print sb[i,0,:]
        plt.plot(contour[i, 0, 0], contour[i, 0, 1], 'ro')
    plt.show()

def contourCentre(contour):
    # Note. input a contour and return the centre of this contour
    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        c = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
    else:
        if len(contour) == 1:
            c = tuple(contour.squeeze().tolist())
        else:
            points = contour.squeeze().tolist()
            c = findMidpoint(*points)
    return c


def contourAngle(contour):
    if len(contour) == 1:
        return
    else:
        moments = cv2.moments(contour)
        mu20 = moments["mu20"] / moments["m00"]
        mu02 = moments["mu02"] / moments["m00"]
        mu11 = moments["mu11"] / moments["m00"]
        if (mu20-mu02) != 0:
            theta = 0.5 * math.atan(2*mu11/(mu20-mu02))
        else:
            theta = math.pi / 2
        return theta


def findSwimBladder(contours):
    # Note. based on the distance between the contour centres
    cs = [contourCentre(cnt) for cnt in contours]
    ds = [distance(p1, p2) for p1, p2 in zip([cs[0], cs[0], cs[1]], [cs[1], cs[2], cs[2]])]
    shortest_i = ds.index(min(ds))
    sb_i = 2-shortest_i
    return sb_i


def findAllContours(image, thresh):
    # Note. only returns the 3 biggest contours
    white = np.full(image.shape, 255, dtype='uint8')
    inverted = white-image
    threshed = applyThreshold(inverted, thresh, 'binary')
    contours = findContours(threshed)  #new_tag
    internals = contours[:3]
    return internals


# ADJUST THRESHOLDS #


def getThreshold(video, winname, thresh_name, start_val):
    try:
        disp = video.getDisplay(winname)
        threshval = disp.trackbars['thresholds'][thresh_name]   #tag
    except ValueError:
        threshval = start_val
    return threshval


def displayCropImage(image, **kwargs):
    if kwargs['roi'] is not None:
        img = cropImage(image, kwargs['roi'])
    else:
        img = image
    return image

def displayMouth(image, **kwargs):

    # Q. displayMouth for each frame?
    # A. yes!

    mouth_thresh = getThreshold(kwargs['video'], kwargs['winname'], kwargs['thresh_name'], kwargs['start_val'])   #tag

    #mouth_thresh = kwargs['mouth_thresh']   ########30 is roughly a good one
    general_thresh = kwargs['general_thresh']

    if kwargs['roi'] is not None:
        img = cropImage(image, kwargs['roi'])
    else:
        img = image

    #switch from dynamic to static
    p1 = (int(kwargs['mouth_roi'][0][0]), int(kwargs['mouth_roi'][0][1]))
    p2 = (int(kwargs['mouth_roi'][1][0]), int(kwargs['mouth_roi'][1][1]))

    cv2.rectangle(img, p1, p2,  (255,0,0) )
    mouth = img[p1[1]:p2[1]+1, p1[0]:p2[0]+1]
    internals = findAllContours(mouth, thresh=mouth_thresh)
    area = cv2.contourArea(internals[0])

    outline = drawContours(mouth, internals, c=10, thresh=mouth_thresh, t=-1)  # new_tag
    # thresh1 = cv2.threshold(mouth, mouth_thresh, 0, cv2.THRESH_BINARY)[1]#########tag
    img[p1[1]:p2[1] + 1, p1[0]:p2[0] + 1] = outline
    outline = cv2.resize(outline, None, fx=16, fy=16, interpolation=cv2.INTER_CUBIC)

    # just return img if user want to see the dynamic position of mouth roi
    return img

def displayThreshold(image, **kwargs):
    # image = self.displayFunction(image, **self.displayKwargs)

    if kwargs['roi'] is not None:
        img = cropImage(image, kwargs['roi'])
    else:
        img = image

    thresh = getThreshold(kwargs['video'], kwargs['winname'], kwargs['thresh_name'], kwargs['start_val'])   #tag
    # Q. means as long as the trackbar's name fits, then it should be ok?
    internals = findAllContours(img, thresh=thresh)

    # sb_i = findSwimBladder(internals)
    # internals.pop(sb_i)

    c_1, th, l_c, l_th, r_c, r_th, size_sb = frameData(img, thresh)  # tag
    outline = drawContours(img, internals, c=255, thresh = thresh, c_1 = c_1, l_c= l_c, r_c=r_c)  #tag

    show = outline.copy()
    #show = cv2.cvtColor(img, cv2.cv.CV_GRAY2BGR)
    blue = (255, 0, 0)
    green = (0, 255, 0)

    drawCCWRotation(show, l_c, l_th, 50, blue)
    drawCCWRotation(show, r_c, r_th, 50, green)

    return show


def mouth_analysis(video, general_thresh=200, initial = 220, mouth_roi=None, general_roi=None, mouth_thresh = 30):

    winname = 'mouth protrusion analysis'
    thresh_name = 'mouth_thresh'

    frame_1st = video.grabFrameN(0)
    c, th, l_c, l_th, r_c, r_th, size_sb = frameData(frame_1st, general_thresh)  # tag
    mid_eyes = findMidpoint(l_c,r_c)
    c_roi = findMidpoint(mouth_roi[0],mouth_roi[1])
    cali_vector = vector(mid_eyes,c_roi)  #c_roi minus mid_eyes

    displayKwargs = dict(video=video, winname=winname, thresh_name=thresh_name, start_val=initial,mouth_thresh = mouth_thresh,
                         roi=general_roi, general_thresh = general_thresh, mouth_roi = mouth_roi, cali_vector = cali_vector)

    video.addDisplay(winname, displayFunction=displayMouth, displayKwargs=displayKwargs)   #tag
    # Just remember: image = self.displayFunction(image, **self.displayKwargs)

    k = cv2.waitKey(0)

    if k == 'q':
        video.removeDisplay(winname)
        return  'good job'
    else:
        video.removeDisplay(winname)
        return 'also good job'

def setmouthThreshold(video, initial, roi, mouth_roi, thresh):

    winname = 'press enter to set mouth protrusion threshold'
    thresh_name = 'mouth_thresh'

    #displayKwargs = dict(video=video, winname=winname, thresh_name=thresh_name, start_val=initial, roi=roi)
    displayKwargs = dict(video=video, winname=winname, thresh_name=thresh_name, start_val=initial,
                         roi=roi, general_thresh=thresh, mouth_roi=mouth_roi)

    video.addDisplay(winname, displayFunction=displayMouth, displayKwargs=displayKwargs)  # tag
    # Just remember: image = self.displayFunction(image, **self.displayKwargs)
    video.addThreshbar(winname, thresh_name, initial)  #tag
    k = cv2.waitKey(0)

    if k == enter_key:
        thresh = getThreshold(video, winname, thresh_name, 'not used')
        video.removeDisplay(winname)
        return thresh
    else:
        return initial



def setThreshold(video, initial, roi):

    winname = 'press enter to set new threshold'
    thresh_name = 'thresh'

    displayKwargs = dict(video=video, winname=winname, thresh_name=thresh_name, start_val=initial, roi=roi)

    video.addDisplay(winname, displayFunction=displayThreshold, displayKwargs=displayKwargs) #tag
    video.addThreshbar(winname, thresh_name, initial)  #tag
    k = cv2.waitKey(0)

    if k == enter_key:
        thresh = getThreshold(video, winname, thresh_name, 'not used')
        video.removeDisplay(winname)
        return thresh
    else:
        return initial


# ANALYSIS FUNCTIONS #


def longAxisAngle(contour, theta):
    phi = contourAngle(contour) # POSITIVE ANGLES ARE CCW IN IMAGE
    v1 = angle2vector(phi)
    v2 = np.array([-v1[1], v1[0]]) # ROTATED 90 DEGREES MORE CCW

    vectors = [v1, v2]

    vx, vy, x0, y0 = cv2.fitLine(contour, distType=cv2.cv.CV_DIST_L2, param=0, reps=0.01, aeps=0.01)
    vc = np.array([vx, vy]).squeeze()

    dot_products = [abs(np.dot(vc, v)) for v in vectors]
    i = dot_products.index(max(dot_products)) # INDEX OF VECTOR PARALLEL WITH LONG AXIS
    # IF ORTHOGONAL VECTOR PARALLEL WITH LONG AXIS, ROTATE PHI 90 DEGREES MORE CCW
    if i == 1:
        phi += math.pi / 2

    vf = angle2vector(theta)
    dot_sign = np.sign(np.dot(vectors[i], vf)) # ORIENTATION OF EYE VECTOR RELATIVE TO BODY AXIS
    # IF SIGN IS NEGATIVE, EYE VECTOR IS OBTUSE WITH BODY AXIS: ROTATE PHI 180 DEGREES
    if dot_sign == -1:
        phi += math.pi

    return phi % (2 * math.pi)


def frameData(image, thresh):
    # c, th, l_c, l_th, r_c, r_th = frameData(img, thresh)

    contours = findAllContours(image, thresh=thresh)

    sb_i = findSwimBladder(contours)   #tag

    sb = contours.pop(sb_i)

    # dt_plot_contour(sb)    # dt_function: the funciton can plot the controur, arguement is a numpy.ndarray kind of coutour

    c = contourCentre(sb)   #tag
    # c is tuple
    # each contour, like sb, is <type 'numpy.ndarray'>

    size_sb = cv2.contourArea(sb)
    # cv2.contourArea returns float

    eye_cs = [contourCentre(eye) for eye in contours]
    eye_c_xs, eye_c_ys = zip(*eye_cs)
    mp = findMidpoint(*eye_cs)

    orientation = angleAB(c, mp) # POSITIVE ANGLES ARE CCW IN IMAGE

    # FOR A CCW ROTATION OF THE BODY AXIS
    if math.pi / 4 <= orientation < 3 * math.pi / 4:
        # DOWN
        left_i = eye_c_xs.index(max(eye_c_xs))
    elif 3 * math.pi / 4 <= orientation < 5 * math.pi / 4:
        # LEFT
        left_i = eye_c_ys.index(max(eye_c_ys))
    elif 5 * math.pi / 4 <= orientation < 7 * math.pi / 4:
        # UP
        left_i = eye_c_xs.index(min(eye_c_xs))
    else:
        # RIGHT
        left_i = eye_c_ys.index(min(eye_c_ys))

    eye_l = contours.pop(left_i)
    eye_l_c = eye_cs.pop(left_i)
    eye_l_th = longAxisAngle(eye_l, orientation)

    eye_r = contours.pop()
    eye_r_c = eye_cs.pop()
    eye_r_th = longAxisAngle(eye_r, orientation)

    '''DT modification module
    a = int(dt_slope_2p(eye_l_c,eye_r_c))
    d = dict()
    d_distance = dict()

    for i in range(sb.shape[0]):
        point = sb[i,0,:]
        # point is actually <type 'numpy.ndarray'>
        b = int(point[1]-point[0]*a)
        if b in d.keys():
            d[b].append(point)
        else:
            d[b] = [point]

    for key in d.keys():
        if len(d[key]) >= 2:
            for i in range(len(d[key])):
                if i == 0:
                    pass
                else:
                    for index in range(i):
                        tem_d = distance(d[key][index], d[key][i])
                        # distance function returns a int...
                        if tem_d in d_distance.keys():
                            d_distance[tem_d].append([d[key][index],d[key][i]])
                        else:
                            d_distance[tem_d] = [[d[key][index],d[key][i]]]

    keys_list = sorted(d_distance)
    # new_sb_c = findMidpoint(d_distance[len(keys_list)-1][0][0],d_distance[len(keys_list)-1][0][1])
    # returned new_sb_c is going to be a tuple
    '''



        ################# how to add them in dictionary!!!!


    return c, orientation, eye_l_c, eye_l_th, eye_r_c, eye_r_th, size_sb


# CHECK TRACKING HELPERS #


def drawCCWRotation(image, p, angle, size, color):
    v = angle2vector(angle)
    v *= size
    p1 = (int(round(p[0])), int(round(p[1])))
    p2 = (int(round(p[0] + v[0])), int(round(p[1] + v[1])))
    cv2.line(image, p1, p2, color=color)
    cv2.circle(image, p1, 3, color, -1)


def showEyes(image, thresh, roi):
    # video.addDisplay(winname, displayFunction=showEyes, displayKwargs={'thresh': thresh, 'roi': roi})  #tag
    print 'enter showEyes'
    if roi is not None:
        print 'did the crop? the thresh is: ', thresh, roi
        img = cropImage(image, roi)
    else:
        img = image

    c, th, l_c, l_phi, r_c, r_phi, size_sb = frameData(img, thresh)

    show = image.copy()
    #show = cv2.cvtColor(img, cv2.cv.CV_GRAY2BGR)
    blue = (255, 0, 0)
    green = (0, 255, 0)

    drawCCWRotation(show, l_c, l_phi, 50, blue)
    drawCCWRotation(show, r_c, r_phi, 50, green)

    return show


