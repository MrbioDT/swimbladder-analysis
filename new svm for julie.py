from svm import *
import numpy as np
from plotSVM import plotSVM
import os
import shelve

if __name__ == '__main__':
    # DT-Code
    # demo = shelve.open('demo_name.shv')
    # print 'demo_name.shv is: ', demo
    # demo.close()
    print 'current working directory is: ', os.getcwd()

    '''PART-1. CONVERT SHV TO BOUTS AND LOAD IT'''
    bout_thresh = .17
    # Question. meaning? how to adjust it?
    # Note. STAR. PARAMETER. Change the threshold of bouts extraction
    # Note. used in load_bouts function
    # Note. it is used here: boutedges, var = extractbouts(angles, threshval) in shvstobouts function
    # Note. more specifically, bouts = ultrasmoothvar>thresh


    behavior_types = ['prey', 'spon']

    bouts = load_bouts('newsvm.pkl', 'C:\\SVM\\SVM training\\', behavior_types, display=0,rebuild_shv=0, threshval = bout_thresh)  #tag
    #  svm_input, svm_input_labels, metrics_list, metrics_index, behavior_types = build_SVM_input(bouts, behavior_types)  #tag
    #  boutfits = [i['tail'] for i in bouts]
    #  for i, tailfit in enumerate(boutfits):
    #    svm_input[i, :] = calculate_metrics(tailfit,metrics_list)

    # DT-Code. print 'so far so good, the bout is : ', bouts
    # Note. Load bouts now, shvs converted to bouts, which is a list stores dictionary now, right?

    bouts = [bout for bout in bouts if bout['vidname']!='']



    # Note. so this step select several items in bouts list, making it a smaller list.
    # Note. 'vidname':video, is one of the pairs in bout dictionary, video should be the key for the shvs file
    # Note. SO THERE MUST BE A PRE-PROCESSING STEP!
    # Note. Julie said this is where they chop the video into each bouts (Obviously, I don't think that's right)
    # debug-Note. bouts is none here
    # DT-Code. print 'bouts: ', bouts

    '''PART-2. BUILD SVM INPUT'''
    svm_input, svm_input_labels, metrics_list, metrics_index, behavior_types = build_SVM_input(bouts, behavior_types)  #tag
    #  boutfits = [i['tail'] for i in bouts]
    #  for i, tailfit in enumerate(boutfits):
    #    svm_input[i, :] = calculate_metrics(tailfit,metrics_list)

    # Note. Bugs because builf_SVM_input didn't give any result!!!
    # DT-Code
    # print 'svm_input: ', svm_input
    # print 'svm_input_labels: ', svm_input_labels

    print [(behavior_types[btype], list(svm_input_labels).count(btype)) for btype in set(svm_input_labels)]

    ix=[np.isnan(svm_input[i,:].max()) for i in range(svm_input.shape[0])]
    svm_input = np.delete(svm_input,np.where(ix),0)
    svm_input_labels = np.delete(svm_input_labels,np.where(ix),0)
    # Note. seems to delete some un-related Metrics?

    def plotmetric(idx):
        temp = svm_input[:, idx].copy()
        slabels = svm_input_labels[temp.argsort()]
        temp.sort()
        pylab.scatter(range(len(temp)) ,temp , c=slabels, lw=0);pylab.show()

    svmc = 1
    svmgamma = .01

    '''PART3-OPTIMIZE SVM METRICS AND GET THE FINAL RESULT'''
    results = optimize_SVM_metrics(svm_input, svm_input_labels, metrics_list, metrics_index, svmc=svmc, svmgamma=svmgamma)   #tag
    # Note. results+=[round_results]
    # Question. But ideally I only want the final round, right?
    label_dict = dict(zip(range(len(behavior_types)),behavior_types))
    results.reverse()
    # Note. all analysis should be done here.


    '''PART4-SHOW THE RESULT'''
    # Question. the following plot for what?
    pylab.figure(figsize=[4, 4], frameon=False)
    pylab.plot(range(1, 1+len(results)), [i[0]['percent'] for i in results], lw=1, marker='o', ms=6)
    pylab.xlabel('# metrics')
    pylab.ylabel('Cross-validated Accuracy')
    pylab.title('SVM Accuracy')
    pylab.ylim([.83, 1])
    pylab.xlim([.09, len(results)])

    def percentformat(x, pos=0):
        return '%1.0f%%' % (100*x)

    pylab.gca().yaxis.set_major_formatter(pylab.matplotlib.ticker.FuncFormatter(percentformat))
    metric_num = 5
    pylab.axvline(metric_num, color='b', linestyle='--')
    pylab.tight_layout()
    pylab.show()

    print results[metric_num-1][0]
    print results[metric_num-1][0]['metrics']
    print [i.__doc__ for i in results[metric_num-1][0]['metrics']]

    index = metric_num-1
    features = [metrics_index[metric] for metric in results[index][0]['metrics']]
    svc = crossvalidatedSVM(svm_input[:, features], svm_input_labels, svmc, svmgamma)[0]
    # return svc.fit(svm_input,svm_input_labels), scores.mean()
    human_label_dict = dict(zip(range(len(behavior_types)), ['Prey Capture', 'Spontaneous Behavior']))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('test',['teal','purple'])
    human_labels = [i.__doc__ for i in results[metric_num-1][0]['metrics']]

    print human_labels

    '''QUESTION HERE!'''
    plotSVM(svc, svm_input[:, features], svm_input_labels, human_labels, ['Prey Capture', 'Spontaneous Behavior'], cmap=cmap)  #tag
    # Questoin. what does this final long function do, man!?