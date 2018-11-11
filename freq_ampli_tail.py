# THIS CODE WILL COMPUTE THE TAILBEAT FREQUENCY
# I combined the functions from Julie's tailfitting analysis
# and add some simple computations to get the tail beat frequency which I defined as
# the ratio between the number of tail oscillations, based on the peak angles of the tail movements of each bout, and duration of the bout
#
# ---------------- USER INPUTS ---------------------
#
# input_path = is the directory path of the shelves from tailfit
# threshval = threshold for selecting bouts from 0.0 - 1.0, the higher the value the more bouts, 0.40 is the default value according to the tailfitting code
# Fs = sampling rate of the video in Hz
# peakthres = the threshold for detecting the tail oscillations, from 0.0 to 20.0, lower value means more peaks, 4 is the recommended value
#
# ----------------OUTPUTS------------------------
#
# [1] tailfreq are the tailbeat frequencies of the bouts
# [2] boutangles are the angles for each frame in the bouts
# [3] peaks are the peaks detected in the plot of computed angles as a function of frame
#
# -------------Example Code-----------------------
# input_path = 'D:\\Tailfits\\'
# threshval=.40
# Fs = 300
# peakthres = 4
#
# freq = tailbeatfreq(input_path,threshval,Fs,peakthres)
# # freq[1][n] contains the frequencies of the nth bout
# # freq[2][n] contains the angles of the nth bout
# # freq[3][n] contains the x,y coordinates of each peak of the nth bout
#
# plt.plot(freq[1][1]) # plot the angles of the 1st bout
# plt.scatter(np.array(freq[2][1][0])[:,0],np.array(freq[2][1][0])[:,1])
# -------------------------------------------------
# Last update: 26 APR 2018, Ivan

import numpy as np
import os
import shelve
from bouts import *
from framemetrics import *
import tailmetrics
import matplotlib.pyplot as plt
import peakdetector


def tailbeatfreq(input_path, threshval, Fs, peakthres, shv_file = None):
    # DT added shv_file is the file name of the shv that need to be analyzed, so that the function can go through one shv for each time

    #the following modification enables this function to read single shv
    if shv_file != None:
        shvs = [str(input_path+'\\'+shv_file)]
        bouts = []
        video_list = []
    else:
        ### Load shv files
        shvs = os.listdir(input_path)
        shvs = [os.path.join(input_path, shv) for shv in shvs if os.path.splitext(shv)[1] == '.shv']
        bouts = []
        video_list = []

    '''LOAD TAIL-FIT'''
    for shvname in shvs:
        # Note. Accroding to the code, shvs should be the dictionary that each value(shvlist) is a tuple of shelve
        shv = shelve.open(shvname)
        print 'Currently analyzing ' + str(shvname)
        # Note. shvs are opened here!

        for i in [1]:
            # Note. useless for loop, added to complete the code

            for video in shv.keys():
                video_list.append(video)
                # Note. shelf should contains several tailfit results for different videos
                # Note. shv[str(os.path.basename(videopath))]=result, according to the tailfit, video/key of shv should be the videopath
                # Note. the value of the shv should be tailfitresult, because IT has 'tailfit' as attribute.  result =  tailfitresult(fittedtail, str(os.path.basename(videopath)),videopath, startpoint, FPS, len(fittedtail), direction, shahash, __version__)

                if type(shv[video]) is list:
                    # Question. how could the key be a list?
                    print 'enter the 1st if, because type(shv[video]) is list'
                    tailfit = shv[video][0]
                else:
                    tailfit = shv[video].tailfit
                    # Note. in such scheme, then the shv[video] should definitely be a tailfitresult class, because it has tailfit attribute!
                    # Note. tailfit correspond to fitted_tail, which is the following:
                    # NOTE. CONFIRMED. fitted_tail is the final result of the whole program.
                    # NOTE. CONFIRMED. it is a list, storing arrays with the total number of total frame analyzed/read
                    # NOTE. CONFIRMED. each array corresponds to one frame, storing x number of points (x=tail_lengh/count)
                    # NOTE. CONFIRMED. points is the fitted result_point(mid point of tail edge), it is the coordinate in each frame
                    # NOTE. CONFIRMED. count would be the final number of total circles.

                '''NORMALIZE & PROCESS TAILFIT '''
                lens = np.array([len(i) for i in tailfit])
                ### Task. I made a big modification... maybe I should do with display as well to see if the threshold is proper

                if lens.var() < 4 and lens.mean() > 3:  # ensure tailfit quality is good
                    # Note. only do the normalization when the tailfit is good ....
                    # Note. lens.var(), Compute the variance along the specified axis.
                    # STAR-Question. count/tail_length has to be bigger than 30? resolution limit?
                    # Question. in theory, tail_length should be the same for each frame, right?

                    if all(lens > 2):
                        tailfit = normalizetailfit(tailfit)
                        # plt.plot(tailfit[0][:,0],tailfit[0][:,1],'b')
                        # plt.plot(tailfit[700][:,0],tailfit[700][:,1],'r')
                        # plt.show()
                        # Question. What if not normalization. For many estimators, including the SVMs, having datasets with unit standard deviation for each feature is important to get good prediction.
                        angles = tail2angles(tailfit)
                        # Note. Calculate the tail_angle!
                        # Note. this function takes tailfit result, for each frame calculate the vector from the startpoint of fitting to the mean point of fraction of tail end
                        # Note. extract the angles of vectors and store the angle of each frame in the returned list.

                        '''PLOT AND DIVIDE THE BOUTS'''
                        boutedges, var = extractbouts(angles, threshval)  # tag
                        print 'the frame range for all the bouts are: ' + str(boutedges)
                        # Note. boutedges are the list storing tuples corresponds to edges of each bout

                        for bout in boutedges:
                            # if boutacceptable(tailfit[bout[0]:bout[1]]):  # tag
                            # task. not really sure how this boutacceptable work, delete if first...
                            bouts += [{'tail': tailfit[bout[0]:bout[1]], 'shvname': shvname, 'vidname': video,
                                       'frames': [bout[0], bout[1]]}]
                            # Note. so the bouts I got here should be ... the list contains all the bouts from all shv files...
                            # Note. the bouts edge info is the value of 'frames'
        tailfreq = []
        boutangles = []
        peaks = []
        Fs = 1 / float(Fs)
        for i in range(len(bouts)):
            nFrames = len(bouts[i]['tail'])
            boutangle = tail2angles(bouts[i]['tail'])  # extract the tailfits of the bout frames
            peak = peakdetector.peakdetold(boutangle, peakthres)  # get the number of peaks which tells us about how many tail beats for the bout
            peak_new = []
            for item in peak[0]:
                peak_new.append([boutedges[i][0]+item[0],item[1]])
            tailfreq.append(len(peak[0]) / float((Fs * nFrames)))
            boutangles.append(boutangle)
            peaks.append(peak_new)

    return tailfreq, boutangles, peaks, boutedges, video_list

if __name__ == '__main__':
    input_path = 'G:\DT-data\\2018\May\May 16\\1st_1st recording'
    threshval = .2
    Fs = 300
    peakthres = 4

    freq = tailbeatfreq(input_path, threshval, Fs, peakthres)
    # # freq[0][n] contains the frequencies of the nth bout
    # # freq[1][n] contains the angles of the nth bout
    # # freq[2][n] contains the x,y coordinates of each peak of the nth bout
    # # freq[3][n] contains the framerange of each bouts
    # # freq[4][n] contains the name of the nth video within that shvs (shvname_list)

    for i, frequency in enumerate(freq[0]):
        print 'Mean tail bend frequency of ', i, 'th bout are: ', frequency, 'Hz'
        print 'Max tail bend angle(abs value) within the ', i, 'th bout are: ', max(
            [abs(x) for x in freq[1][i]]), 'degree'
        framerange = range(freq[3][i][0],
                           freq[3][i][1])  # construct the framerange of each bouts and use it as x for plotting
        plt.plot(framerange, freq[1][i])  # plot the angles of the 1st bout
        # plt.scatter(np.array(freq[2][i])[:, 0], np.array(freq[2][i])[:, 1])
        # don't want to debug the first one... anyway I got the plotting
        plt.savefig(input_path + '\\' + str(freq[4][0]) + '_' + str(
            i) + 'th_bout.png')  # assume there is only one video in the shv file
        plt.clf()

