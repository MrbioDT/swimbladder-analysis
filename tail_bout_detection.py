import numpy as np
import os
import pickle
from sklearn import cross_validation, svm, preprocessing
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
import shelve
from bouts import *
from framemetrics import *
import tailmetrics
from injectargs import *
from plotSVM import *
import matplotlib.pyplot as plt


### User input
input_path = 'C:\DT files\Julie Semmelhack Lab\python learning\code\Joe Tailfit 0.9.3 (with DT note) Vol.2-bout detection' #the folder that contains all the shv data
threshval=.40   # threshold value to detect the bouts, usually used .17???

### Load shv files
shvs = os.listdir(input_path)
shvs = [os.path.join(input_path, shv) for shv in shvs if os.path.splitext(shv)[1] == '.shv']
bouts = []

'''LOAD TAIL-FIT'''
for shvname in shvs:
    # Note. Accroding to the code, shvs should be the dictionary that each value(shvlist) is a tuple of shelve
    shv = shelve.open(shvname)
    print 'Currently analyzing ' + str(shvname)
    # Note. shvs are opened here!

    for i in [1]:
    # Note. useless for loop, added to complete the code

        for video in shv.keys():
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
                    print boutedges
                        # Note. boutedges are the list storing tuples corresponds to edges of each bout

                    for bout in boutedges:
                        # if boutacceptable(tailfit[bout[0]:bout[1]]):  # tag
                        # task. not really sure how this boutacceptable work, delete if first...
                        bouts += [{'tail': tailfit[bout[0]:bout[1]], 'shvname': shvname,'vidname': video, 'frames': [bout[0], bout[1]]}]
                            # Note. so the bouts I got here should be ... the list contains all the bouts from all shv files...
                            # Note. the bouts edge info is the value of 'frames'

for i in range(len(bouts)):
    print bouts[i]['frames']
#print bouts[0]['vidname']




### Output to a list contains dictionaryu