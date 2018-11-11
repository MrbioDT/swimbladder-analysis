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



def svm_cgamma(X,Y):
    C_range = 10.0 ** np.arange(-3, 4)
    gamma_range = 10.0 ** np.arange(-4, 3)
    param_grid = dict(gamma=gamma_range, C=C_range)

    cv = StratifiedKFold(y=Y, n_folds=3)
    grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
    grid.fit(X, Y)
    print("The best classifier is: ", grid.best_estimator_)

    # plot the scores of the grid
    # cv_scores_ contains parameter settings and scores
    score_dict = grid.grid_scores_

    # We extract just the scores
    scores = [x[1] for x in score_dict]
    scores = np.array(scores).reshape(len(C_range), len(gamma_range))

    # draw heatmap of accuracy as a function of gamma and C
    pylab.figure(figsize=(8, 6))
    pylab.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.95)
    pylab.imshow(scores, interpolation='nearest', cmap=pylab.cm.spectral, vmin = 0, vmax = 1)
    pylab.xlabel('gamma')
    pylab.ylabel('C')
    pylab.colorbar()
    pylab.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    pylab.yticks(np.arange(len(C_range)), C_range)

    pylab.show()


def shvstobouts(input_path='.', behavior_types=['prey', 'spon', 'escape'], display=False, threshval=.45):
    # Note. should be the place to convert shv into bouts
    # bouts = shvstobouts(shvpath, behavior_types,display=display, threshval = threshval)
    # DT-Code: print 'input_path in shvtobouts is: ', input_path

    '''READ AND BUILD SHV'''
    threshes = []
    # Question. when is the threshes applied?

    shvs = os.listdir(input_path)
    shvs = [os.path.join(input_path, shv) for shv in shvs if os.path.splitext(shv)[1] == '.shv']
    shvs = {behavior: filter(lambda k: behavior in k, shvs) for behavior in behavior_types}  ###########...
    # Question. I need to confirm what information should be contained in shvs? by checking how it is used.
    # Answer. Accroding to the code, shvs should be the dictionary that each value is a tuple of shelve, key would be corresponding behaviors
    # Answer. each shelve should contains tailfit results for videos
    # filter(function, iterable) is equivalent to the generator expression (item for item in iterable if function(item))
    print '3rd shvs in shvstobouts: ', shvs

    colors = dict(zip(behavior_types,['b','g','r','y','c','m','k']))
    bouts = []

    '''LOAD TAIL-FIT'''
    for behavior,shvlist in shvs.iteritems():
    # Note. Accroding to the code, shvs should be the dictionary that each value(shvlist) is a tuple of shelve

        for shvname in shvlist:
            shv = shelve.open(shvname)
            # Note. shvs are opened here!

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
                    # tailfit = normalizetailfit(tailfit)
                        # bouts += [{'tail':tailfit[bout[0]:bout[1]], 'behavior':behavior, 'shvname':shvname,
                        #                   'vidname':video,'frames':[bout[0],bout[1]]}]
                                #  Note. key question is the what is tailfit[bout[0]:bout[1]]?
                                #  bouts = shvstobouts(shvpath, behavior_types,display=display, threshval = threshval )   #DT-tag
                                #  svm_input, svm_input_labels, metrics_list, metrics_index, behavior_types = build_SVM_input(bouts, behavior_types)  #tag
                                #  boutfits = [i['tail'] for i in bouts]
                                #  for i, tailfit in enumerate(boutfits):
                                #    svm_input[i, :] = calculate_metrics(tailfit,metrics_list)


                    # Note. in such scheme, then the shv[video] should definitely be a tailfitresult class, because it has tailfit attribute!
                    # Note. tailfit correspond to fitted_tail, which is the following:
                    # NOTE. CONFIRMED. fitted_tail is the final result of the whole program.
                    # NOTE. CONFIRMED. it is a list, storing arrays with the total number of total frame analyzed/read
                    # NOTE. CONFIRMED. each array corresponds to one frame, storing x number of points (x=tail_lengh/count)
                    # NOTE. CONFIRMED. points is the fitted result_point(mid point of tail edge), it is the coordinate in each frame
                    # NOTE. CONFIRMED. count would be the final number of total circles.

                '''NORMALIZE & PROCESS TAILFIT '''
                lens = np.array([len(i) for i in tailfit])

                if lens.var() < 4 and lens.mean() > 30: #ensure tailfit quality is good
                    # Note. only do the normalization when the tailfit is good ....
                    # Note. lens.var(), Compute the variance along the specified axis.
                    # STAR-Question. count/tail_length has to be bigger than 30? resolution limit?
                    # Question. in theory, tail_length should be the same for each frame, right?

                    if all(lens > 20):
                        tailfit = normalizetailfit(tailfit)
                        # bouts += [{'tail':tailfit[bout[0]:bout[1]], 'behavior':behavior, 'shvname':shvname,
                        #                   'vidname':video,'frames':[bout[0],bout[1]]}]
                                #  Note. key question is the what is tailfit[bout[0]:bout[1]]?
                                #  bouts = shvstobouts(shvpath, behavior_types,display=display, threshval = threshval )   #DT-tag
                                #  svm_input, svm_input_labels, metrics_list, metrics_index, behavior_types = build_SVM_input(bouts, behavior_types)  #tag
                                #  boutfits = [i['tail'] for i in bouts]
                                #  for i, tailfit in enumerate(boutfits):
                                #    svm_input[i, :] = calculate_metrics(tailfit,metrics_list)
                        #plt.plot(tailfit[0][:,0],tailfit[0][:,1],'b')
                        #plt.plot(tailfit[700][:,0],tailfit[700][:,1],'r')
                        #plt.show()
                        # Question. What if not normalization. For many estimators, including the SVMs, having datasets with unit standard deviation for each feature is important to get good prediction.
                        angles = tail2angles(tailfit)
                        # Note. Calculate the tail_angle!
                        # Note. this function takes tailfit result, for each frame calculate the vector from the startpoint of fitting to the mean point of fraction of tail end
                        # Note. extract the angles of vectors and store the angle of each frame in the returned list.

                        '''PLOT AND DIVIDE THE BOUTS'''
                        if display:
                            # Note. display is 0, so will not enter this if in normal situation
                            if threshes:
                                threshinit = threshes[-1]
                            else:
                                threshinit = threshval

                            ibouts = boutplotter(angles, colors[behavior], threshinit, title=os.path.basename(video))
                            boutedges = ibouts.bouts #tag
                            # return bouts in list
                            threshes += [ibouts.thresh]

                        else:
                            boutedges, var = extractbouts(angles, threshval)  #tag
                            # Note. boutedges are the list storing tuples corresponds to edges of each bout

                        for bout in boutedges:
                            if boutacceptable(tailfit[bout[0]:bout[1]]):   #tag
                                bouts += [{'tail':tailfit[bout[0]:bout[1]], 'behavior':behavior, 'shvname':shvname,
                                           'vidname':video,'frames':[bout[0],bout[1]]}]
                                #  Note. key question is the what is tailfit[bout[0]:bout[1]]?
                                #  bouts = shvstobouts(shvpath, behavior_types,display=display, threshval = threshval )   #DT-tag
                                #  svm_input, svm_input_labels, metrics_list, metrics_index, behavior_types = build_SVM_input(bouts, behavior_types)  #tag
                                #  boutfits = [i['tail'] for i in bouts]
                                #  for i, tailfit in enumerate(boutfits):
                                #    svm_input[i, :] = calculate_metrics(tailfit,metrics_list)

                                # Question. [bout[0]:bout[1] what does this means?    #####tag
    return bouts


def calculate_metrics(tailfit, metric_list):
    metrics = np.zeros(len(metric_list))
    for i, metric in enumerate(metric_list):
        metrics[i] = metric(tailfit)
    return metrics


def crossvalidatedSVM(svm_input, svm_input_labels, C=1, gamma=.01, kernel='rbf'):
    """takes in a metric list and generates a crossvalidated SVM"""
    # Note. svc_result, correct = crossvalidatedSVM(svm_input, svm_labels, svmc, svmgamma)

    svc=svm.SVC(kernel=kernel, C=C, gamma=gamma)
    # Note. C-Support Vector Classification.
    # DT-Code. print 'svm_input_labels: ', svm_input_labels

    cv = cross_validation.StratifiedKFold(y=svm_input_labels, n_folds=5)
    # Question. the type of cv?
    # Question. the principle that how StratifiedKFold works?
    # Note. Stratified K-Folds cross validation iterator, Provides train/test indices to split data in train test sets.
    # Note. bugs exist because svm_input_labels is None.
    # ValueError: Cannot have number of folds n_folds=5 greater than the number of samples: 0.

    scores = cross_validation.cross_val_score(svc, svm_input, svm_input_labels, cv=cv)
    # Note. Evaluate a score by cross-validation
    # Note.

    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
    return svc.fit(svm_input,svm_input_labels), scores.mean()
    # Question. why svc can be fitted into data?
    # Question. what is the type of svc.fit
    # Note. fit(), Fit the SVM model according to the given training data. return self (object)


def load_bouts(filepath='bouts.pkl', shvpath='.\\shvs\\', behavior_types=['prey', 'spon', 'escape'], display=False, rebuild_shv=False, threshval=.45):
    # Note. Load the shvs files and convert into bouts
    # DT-Code. print 'shvpath in function load_bouts: ', shvpath
    # DT-code. print 'filepath in function load_bouts: ', filepath
    # bouts = load_bouts('newsvm.pkl', 'C:\\SVM\\SVM training\\', behavior_types, display=0,rebuild_shv=0, threshval = bout_thresh)
    # bout['tail'] is the type of data I am looking for...

    if os.path.exists(filepath) and not rebuild_shv:
        # Question. underwhat circumstance should we load data from pre-exist pkl files?
        try:
            with open(filepath,'rb') as f:
                bouts = pickle.load(f)    #tag
                # Question. where does the bouts.pkl comes from
        except:
            print "Failed to load from pickle"
    else:
        if os.path.exists(filepath):
            os.unlink(filepath)

        bouts = shvstobouts(shvpath, behavior_types,display=display, threshval = threshval )   #DT-tag
        #  svm_input, svm_input_labels, metrics_list, metrics_index, behavior_types = build_SVM_input(bouts, behavior_types)  #tag
        #  boutfits = [i['tail'] for i in bouts]
        #  for i, tailfit in enumerate(boutfits):
        #    svm_input[i, :] = calculate_metrics(tailfit,metrics_list)


        with open(filepath, 'wb') as f:
            # Note. 'wb' mode can help to create a pickle file.
            pickle.dump(bouts, f)
            # Note. Write a pickled representation of obj to the open file object file. This is equivalent to Pickler(file, protocol).dump(obj).
            # Question. what does pickled means?
    return bouts


class SVMPredict(object):
    @injectArguments
    def __init__(self, svc=None, metric_list=None, label_dict=None, bout_thresh=.15):
        pass

    @classmethod
    def loadfromshv(cls, shv='svm.shv'):
        shv = shelve.open(shv)
        svc = shv['svm_predict']
        shv.close()
        return svc

    def savetoshv(self, shvpath='svm.shv'):
        shv = shelve.open(shvpath)
        shv['svm_predict'] = self
        shv.close()

    def predictbout(self, tailfit_slice):
        assert tailfit_slice
        return self.svc.predict(calculate_metrics(tailfit_slice, self.metric_list))

    def predict(self, tailfit):
        assert self.svc and self.metric_list and self.label_dict

        normtailfit = normalizetailfit(tailfit)
        angles = tail2angles(normtailfit)
        #TODO check for shoddy tailfits, using standard tailfit func

        bouts = extractbouts(angles, self.bout_thresh)[0]
        bouts =  [bout for bout in bouts if boutacceptable(tailfit[bout[0]:bout[1]])]
        results = []
        for bout in bouts:
            try:
                results += [{'bout':bout,'behavior':self.label_dict[int(self.predictbout(normtailfit[bout[0]:bout[1]]))]}]
            except:
                results += [{'bout':bout,'behavior':"FAILED"}]
        return results

    def predictmany(self, tailfitlist):
        return [self.predict(tailfit) for tailfit in tailfitlist]


def build_SVM_input(bouts, behavior_types=['prey', 'spon', 'escape'], norm=False):
    # svm_input, svm_input_labels, metrics_list, metrics_index, behavior_types = build_SVM_input(bouts, behavior_types)

    '''BUILD INPUTS'''
    labels = dict(zip(behavior_types,range(len(behavior_types))))
    boutfits = [i['tail'] for i in bouts]
    #  for i, tailfit in enumerate(boutfits):
    #    svm_input[i, :] = calculate_metrics(tailfit,metrics_list)
    behaviors = [i['behavior'] for i in bouts]
    # Note. bouts += [{'tail':tailfit[bout[0]:bout[1]], 'behavior':behavior, 'shvname':shvname,'vidname':video,'frames':[bout[0],bout[1]]}] in load_bouts()

    metrics_list = tailmetrics.metric_list   #tag
    # Question. how does this function takes input? should be a tailfit? and how does it gives out output?
    # Note. metrics_list is goint to be a list stores 16 parameters (hashed? not sure)
    # DT-Code. print 'metrics_list in build_SVM_input: ', metrics_list

    svm_input = np.zeros((len(boutfits), len(metrics_list)))

    for i, tailfit in enumerate(boutfits):
        svm_input[i, :] = calculate_metrics(tailfit,metrics_list)

    svm_labels = np.array([labels[i] for i in behaviors])
    metrics_index = dict(zip([i for i in metrics_list], range(len(metrics_list))))

    '''NORMALIZATION'''
    if norm:
        svm_input = preprocessing.normalize(svm_input)
        svm_input = preprocessing.scale(svm_input)

    return svm_input, svm_labels, metrics_list, metrics_index, behavior_types

def optimize_SVM_metrics(svm_input, svm_labels, metrics_list, metrics_index, svmc=1.0, svmgamma=.01):
    """Optimizes a set of SVM metrics by leaving one out each round (the metric without which the svm performed best)"""
    # Note. results = optimize_SVM_metrics(svm_input, svm_input_labels, metrics_list, metrics_index, svmc=svmc, svmgamma=svmgamma)

    results = []

    for number_metrics in list(reversed(range(1,len(metrics_list)+1))):
        # Note. start from 16 to 1
        round_results = []

        if number_metrics == len(metrics_list):
            svc_result, correct = crossvalidatedSVM(svm_input, svm_labels, svmc, svmgamma)    #tag
            # return svc.fit(svm_input,svm_input_labels), scores.mean()
            round_results += [{'metrics_name': [m.func_name for m in metrics_list],
                                   'metrics':metrics_list,
                                   'percent':correct, 'drop_metric':None, 'svc':svc_result}]
        else:
            for i in range(len(metrics_list)):
                svm_input_loo = svm_input[:, [metrics_index[metric] for j, metric in enumerate(metrics_list) if i!=j]]
                svc_result, correct = crossvalidatedSVM(svm_input_loo, svm_labels, svmc, svmgamma)
                round_results += [{'metrics_name': [metrics_list[j].func_name for j in range(len(metrics_list)) if j != i],
                                   'metrics':[metrics_list[j] for j in range(len(metrics_list)) if j != i],
                                   'percent':correct, 'drop_metric':metrics_list[i].func_name, 'svc':svc_result}]

            round_results.sort(key=lambda t: t['percent'], reverse=True)
            # Note. choose the round_results with the max correct percent

        results+=[round_results]
        print "Best accuracy: ", round(round_results[0]['percent'], 3), '%  metrics: ', round_results[0]['metrics_name']
        print "Dropping: ", round_results[0]['drop_metric']
        metrics_list = round_results[0]['metrics']
    return results


