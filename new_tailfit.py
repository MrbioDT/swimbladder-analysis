from filepicker import *  # SQ*
from eye_tracker_helpers import *
import pandas as pd
import os
from matplotlib import pyplot as plt
import csv
import copy
from autodetection_tools import *
from tailfit9 import *
from freq_ampli_tail import *

if __name__ == "__main__":

    ### USER INPUT HERE ###
    # folder = 'G:\DT-data\\2018\May\May 16\\1st_1st recording'  # this should be the folder where you put all the videos for analysis # folder = pickdir() # alternative way to do

    folder = 'D:\DATA\\22 APR 2018\E008\A1\\tail'
    tail_fitting = 'y'  # y for doing the tailfitting
    tail_startpoint = None

    # READ ALL THE AVIS FILES WITHIN THAT FOLDER
    filenames = os.listdir(folder)
    avis = [filename for filename in filenames if os.path.splitext(filename)[1] == '.avi']

    for avi in avis:
        print '*****************************************************************************************************************************************************************************************************************************************'
        print 'current processing is: ', avi  # tell the user which avi is processing

        if tail_fitting == 'y':
            # display = askyesno(text='Display frames?')
            display = True
            displayonlyfirst = True

            'TAIL FITTTING'
            video_path = str(folder + '\\' + avi)
            if str(type(tail_startpoint)) == "<type 'NoneType'>":
                # you can either set the startpoint or process the same batch of videos with the startpoint setted in the first videos
                tail_startpoint, tail_amplitude_list = tailfit_batch([video_path], display, displayonlyfirst,
                                                                     shelve_path=folder + '\\' + avi + '.shv',
                                                                     reuse_startpoint=True)
                # tail_ampitude_list actually stores the absolute value for tail movement

            else:
                display = True
                tail_startpoint, tail_amplitude_list = tailfit_batch([video_path], display, displayonlyfirst,
                                                                     shelve_path=folder + '\\' + avi + '.shv',
                                                                     reuse_startpoint=True,
                                                                     tail_startpoint=tail_startpoint)
            # the corresponding shv files will be saved to the same folder as the tailfit
            # tail_ampitude_list actually stores the absolute value for tail movement


            'PLOT IT OUT'
            threshval = .40
            Fs = 300
            peakthres = 4

            max_amplitude_list = []
            freq = tailbeatfreq(folder, threshval, Fs, peakthres, shv_file=avi + '.shv')
            # # freq[0][n] contains the frequencies of the nth bout
            # # freq[1][n] contains the angles of the nth bout
            # # freq[2][n] contains the x,y coordinates of each peak of the nth bout
            # # freq[3][n] contains the framerange of each bouts
            # # freq[4][n] contains the name of the nth video within that shvs (shvname_list)

            for i, frequency in enumerate(freq[0]):
                print 'Mean tail bend frequency of ', i, 'th bout are: ', frequency, 'Hz'
                max_amplitude = max([abs(x) for x in freq[1][i]])
                max_amplitude_list.append(max_amplitude)
                print 'Max tail bend angle(abs value) within the ', i, 'th bout are: ', max_amplitude, 'degree'
                framerange = range(freq[3][i][0],
                                   freq[3][i][1])  # construct the framerange of each bouts and use it as x for plotting

                if len(framerange) != len(freq[1][i]):
                    # when the tail is too close to the edge, fitting will be missing for that frame
                    framerange = range(freq[3][i][0], freq[3][i][0] + len(freq[1][i]))

                plt.plot(framerange, freq[1][i])  # plot the bending angles of the ith bout

                # plt.scatter(np.array(freq[2][i])[:, 0], np.array(freq[2][i])[:, 1])
                plt.savefig(folder + '\\' + avi + '_' + str(freq[4][0]) + '_' + str(
                    i) + 'th_bout.png')  # assume there is only one video in the shv file
                plt.clf()
