from sklearn import  svm
import itertools
import pdb
import numpy as np
import pylab
from sklearn.metrics import confusion_matrix
from bouts import normalizetailfit
from framemetrics import tail2angles, tail2sumangles, tail2tipangles
from matplotlib import gridspec, pyplot
import matplotlib
import pandas

def add_ax_labels(fig=None):
    if fig == None:
        fig = pyplot.gcf()
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i), va="center", ha="center")

def smart_lims(data, factor = .05): #built in auto scale kinda sucks, wastes whitespace
        datarange = np.abs(data.min()- data.max())*factor
        return [data.min() - datarange, data.max() + datarange]

class plotSVM():
    def __init__(self, svm, svm_input, svm_input_labels, metric_names, labels_human_names, bouts=None, colors=None, savename=None, cmap = pylab.cm.OrRd ):
        # plotSVM(svc, svm_input[:, features], svm_input_labels, human_labels, ['Prey Capture', 'Spontaneous Behavior'], cmap=cmap)
        assert len(metric_names) == svm_input.shape[1]
        assert svm_input.shape[0] == len(svm_input_labels)
        if colors:
            assert len(metric_names) == len(colors)
        self.svm_input = svm_input
        self.svm_input_labels = svm_input_labels
        df=pandas.DataFrame(np.hstack([svm_input,svm_input_labels[:,None]]))
        print df.head()

        self.human_labels = metric_names
        self.bouts = bouts
        self.predictions = svm.predict(svm_input)
        # Perform classification on samples in X.
        # y_pred : array, shape (n_samples,). Class labels for samples in X.

        print confusion_matrix(svm_input_labels, self.predictions)   #task. worth spending time, haha
        # sklearn.metrics.confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)
        # Compute confusion matrix to evaluate the accuracy of a classification
        # Question. different between score? and this?
        print metric_names

        fig = pyplot.figure(figsize=(8, 8), dpi=100)
        pylab.axes([.1,.1,.92,.92])
        self.cids = []
        self.xydict = {}

        predictions = svm.predict(svm_input)
        correct_idx = (predictions==svm_input_labels).nonzero()
        incorrect_idx = (predictions!=svm_input_labels).nonzero()

        norm = pylab.Normalize(min(svm_input_labels), max(svm_input_labels))
        incorrect_edgecolors=cmap(norm(predictions[incorrect_idx]))

        nrows = svm_input.shape[1]-1
        gs = gridspec.GridSpec(nrows, nrows, wspace=.08, hspace=.08)
        pyplot.gcf().set_facecolor('white')

        for x, y in itertools.product(range(nrows), repeat=2):
        # Question. what does this for loops do????
            if x <= y:
                self.xydict[pyplot.subplot(gs[y,x])] = [x,y]
                pyplot.tick_params(axis='both', which='both', right='off', top='off')

                if x == 0:
                    pyplot.ylabel(metric_names[y+1])
                    pyplot.gca().yaxis.set_label_coords(-0.25-.14*(y % 2), .5)

                else:
                    pyplot.tick_params(axis='y', labelleft='off')
                if y == nrows-1:
                    pyplot.xlabel(metric_names[x])
                    pyplot.gca().xaxis.set_label_coords(.5, -0.2-.14*(x % 2))

                else:
                    pyplot.tick_params(axis='x', labelbottom='off')

                spines_to_remove = ['top', 'right']
                for spine in spines_to_remove:
                    pyplot.gca().spines[spine].set_visible(False)


                perm = np.random.permutation(svm_input.shape[0])
                pylab.scatter(svm_input[:, x][perm], svm_input[:, y + 1][perm], c=svm_input_labels[perm], cmap=cmap, norm=norm, marker='o', s=30, alpha=.3,  edgecolor=(.3,.3,.3), lw=.6)
                pylab.xlim(smart_lims(svm_input[:, x], .1))
                pylab.ylim(smart_lims(svm_input[:, y+1], .1))

                pyplot.locator_params(tight=True, nbins=6) #this doesn't work so great, not sure why?

        pylab.suptitle("SVM Metric Pairs Projected into 2d", fontsize=14)
        labelrects = [pylab.Rectangle((0, 0), 1, 1, facecolor=cmap(norm(c)), edgecolor='black') for c in range(len(labels_human_names))]
        labeltexts = labels_human_names
        pylab.legend(labelrects, labeltexts, loc='center', bbox_to_anchor=(-1.0, svm_input.shape[1]-1), prop={'size': 12})

        if savename:
            pyplot.savefig(savename, transparent=True)
        pyplot.show()

    def quit_subfigure(self, event):
        if event.key == 'alt+q' or event.key == 'q': #it adds an alt to the keypress and I don't know why
            pyplot.close()

