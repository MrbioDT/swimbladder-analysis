from svm import *
import shelve
import numpy as np


class boutplotter(object): #just for this plot example
    def __init__(self,angles, color='b', threshinit = .45, title = None):
        self.default_span_alpha = .4
        self.angles = angles
        self.color = color
        self.threshinit  = threshinit

        self.figure = pylab.figure(figsize=[4, 4])
        pylab.subplots_adjust(bottom=.05, left=.05, right=.95)
        self.smooth = extractbouts(angles)[1]

        self.spanhandles = []
        self.hlinehandles = []

        pylab.subplot(211)
        pylab.plot(self.angles,self.color, linewidth=2)[0]
        pylab.tick_params(labelbottom = 'off', labelleft='off')
        pylab.subplot(212)
        pylab.plot(self.smooth,'k', figure=self.figure, linewidth=2)
        pylab.tick_params(labelbottom = 'off', labelleft='off')
        self.toggle_clicks = []

        self.update(self.threshinit)
        self.show()

    def close(self):
        self.bouts = self.get_bouts()
        [self.figure.canvas.mpl_disconnect(cid) for cid in self.cids]

        del self.threshslider
        self.clear_handles()
        pylab.close(self.figure)
        del self.figure

    def quit_figure(self, event):
        if event.key == 'alt+q' or event.key == 'q': #it adds an alt to the keypress and I don't know why
            self.close()

    def clear_handles(self):
        try:
            for h in self.spanhandles + self.hlinehandles:
                h.remove()
                del h
        except NameError:
            pass
##            print 'nameerror'
        self.spanhandles = []
        self.hlinehandles = []

    def update(self, val):
        boutedges,var = extractbouts(self.angles,val)
        self.boutedges = boutedges
        self.clear_handles()

        pylab.subplot(211)
        pylab.xlim((0,len(self.angles)))
        self.spanhandles = [pylab.axvspan(i[0],i[1],alpha = self.default_span_alpha,color=self.color,figure=self.figure) for i in boutedges]
        pylab.subplot(212)
        pylab.xlim((0,len(self.angles)))
        [pylab.axvline(i[0],alpha = self.default_span_alpha,color=self.color,figure=self.figure) for i in boutedges]
        [pylab.axvline(i[1],alpha = self.default_span_alpha,color=self.color,figure=self.figure) for i in boutedges]
        self.hlinehandles += [pylab.axhline(val,alpha=1,color='r',figure=self.figure)]

        #this redraws the previously clikced on spans
        clicks = list(self.toggle_clicks) #to ensure a deep copy
        self.toggle_clicks = []
        [self.toggle_click_point(point, True) for  point in clicks]

        pylab.draw()
        self.bouts = self.get_bouts()

    def show(self):
        pylab.show()

    def get_bouts(self):
        return [bout for axv, bout in zip(self.spanhandles, self.boutedges) if not any([axv.contains_point(click) for click in self.toggle_clicks])]

def plot_bouts_fig(input_path, behavior_types, bout_thresh):
    # plot_bouts_fig('C:\\SVM\\SVM training\\', behavior_types, bout_thresh), btw, behavior_types = ['prey', 'spon']

    shvs = os.listdir(input_path)
    shvs = [os.path.join(input_path, shv) for shv in shvs if os.path.splitext(shv)[1] == '.shv']
    # Note. read all the shelve file in the directory

    shvs = {behavior: filter(lambda k: behavior in k, shvs) for behavior in behavior_types}
    # Note. so shvs now contains the the fitted tail movement classified according to behavior
    # Question. when is classification performed?
    # Note. filter(function, iterable) is equivalent to the generator expression (item for item in iterable if function(item))
    # Question. is 'prey', 'spon' contained in the shvs?
    # Answer. I don't think so.
    # Note. Here is what output looks like after tailfit9
    # Note. result =  tailfitresult(fittedtail, str(os.path.basename(videopath)),videopath, startpoint, FPS, len(fittedtail), direction, shahash, __version__)
    # Note. shv[str(os.path.basename(videopath))]=result

    colors = dict(zip(behavior_types,['b','g','r','y','c','m','k']))    ##################tag
    for behavior, shvlist in shvs.iteritems():
        for shvname in shvlist:
            shv = shelve.open(shvname)

            for video in shv.keys():
                if type(shv[video]) is list:
                    tailfit = shv[video][0]
                else:
                    tailfit = shv[video].tailfit
                lens = np.array([len(i) for i in tailfit])
                if lens.var() < 4 and lens.mean() > 30:
                    if all(lens > 20):
                        tailfit = normalizetailfit(tailfit)
                        angles = tail2angles(tailfit)
                        boutplotter(angles, colors[behavior], bout_thresh, title=os.path.basename(video))
                break
            break
        break


if __name__ == '__main__':
    bout_thresh = .17
    behavior_types = ['prey', 'spon']
    plot_bouts_fig('C:\\SVM\\SVM training\\', behavior_types, bout_thresh)   #tag


