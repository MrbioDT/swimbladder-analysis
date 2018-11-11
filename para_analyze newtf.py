from svm import *


if __name__ == '__main__':
    bout_thresh = .17
    behavior_types = ['prey', 'spon']
    bouts = load_bouts('paranewbouts.pkl', 'C:\\SVM\\newpara\\', behavior_types, display=0,rebuild_shv=0, threshval = bout_thresh)
    bouts = [bout for bout in bouts if bout['vidname']!='']

    drop_ix = [not (bout['vidname']!='20-02-54.496.avi' and bout['frames'][0]!=453) for bout in bouts]
    svm_input, svm_input_labels, metrics_list, metrics_index, behavior_types = build_SVM_input(bouts, behavior_types)

    print [(behavior_types[btype], list(svm_input_labels).count(btype)) for btype in set(svm_input_labels)]
    ix=[np.isnan(svm_input[i,:].max()) for i in range(svm_input.shape[0])]

    svm_input = np.delete(svm_input,np.where(ix),0)
    svm_input_labels = np.delete(svm_input_labels,np.where(ix),0)

    svmc, svmgamma = 1, .01
    results = optimize_SVM_metrics(svm_input, svm_input_labels, metrics_list, metrics_index, svmc, svmgamma)
    svm_input = np.delete(svm_input,np.where(drop_ix),0)
    svm_input_labels = np.delete(svm_input_labels,np.where(drop_ix),0)

    label_dict = dict(zip(range(len(behavior_types)),behavior_types))
    results.reverse()
    pylab.figure(figsize=[4, 4], frameon=False)
    pylab.plot(range(1, 1+len(results)), [i[0]['percent'] for i in results], lw=2, marker='o', ms=6)
    pylab.xlabel('# metrics')
    pylab.ylabel('Cross-validated Accuracy')
    pylab.title('Paramecia SVM Accuracy')
    pylab.ylim([.78, 1])
    pylab.xlim([0, len(results)])
    def percentformat(x, pos=0):
        return '%1.0f%%' % (100*x)
    pylab.gca().yaxis.set_major_formatter(pylab.matplotlib.ticker.FuncFormatter(percentformat))

    metric_num = 6
    pylab.axvline(metric_num, color='b', linestyle='--')
    pylab.tight_layout()
    pylab.show()
    print results[metric_num-1][0]
    metrics_order = [[m.__doc__ for m in r[0]['metrics']] for r in results[:metric_num]]
    print [[j for j in i[0] if j not in i[1]] for i in zip(metrics_order,[[]]+metrics_order[:-1])]
    index = metric_num-1

    features = [metrics_index[metric] for metric in results[index][0]['metrics']]
    direct_svc = results[index][0]['svc']

##    svm_cgamma(svm_input[:,[metrics_index[metric] for metric in results[index][0]['metrics']]],svm_labels)
    svc = crossvalidatedSVM(svm_input[:, features], svm_input_labels, svmc, svmgamma)[0]
    human_label_dict = dict(zip(range(len(behavior_types)), ['Prey Capture', 'Spontaneous Behavior']))

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('test',['teal','purple'])
    human_labels = [i.__doc__ for i in results[metric_num-1][0]['metrics']]
    print human_labels
    plotSVM(svc, svm_input[:, features], svm_input_labels, human_labels, ['Prey Capture', 'Spontaneous Behavior'], cmap=cmap)   #tag

    svm_predict = SVMPredict(svc, results[index][0]['metrics'], label_dict, bout_thresh = bout_thresh)