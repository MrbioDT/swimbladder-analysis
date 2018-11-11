import pylab
import scipy.stats
from framemetrics import *
import matplotlib.pyplot as plt

Slider = pylab.matplotlib.widgets.Slider

def interp_nans(data):
    '''Interpolates NaN - modifies in place! Still returns for convenience.
    '''
    mask = numpy.isnan(data) + numpy.isinf(data)
    data[mask] = numpy.interp(numpy.flatnonzero(mask), numpy.flatnonzero(~mask), data[~mask])
    return data

def normalizetailfit(tailfit):
    """Takes in a tailfit, and returns a normalized version which is goes from 0 to 1 normalized to taillength averaged over the first few frames
    """
    # Note. Just remind you what tailfit is:
    # NOTE. CONFIRMED. fitted_tail is the final result of the whole program.
    # NOTE. CONFIRMED. it is a list, storing arrays with the total number of total frame analyzed/read
    # NOTE. CONFIRMED. each correspond to one frame, storing x number of points (x=tail_lengh/count)
    # NOTE. CONFIRMED. points is the fitted result_point(mid point of tail edge), it is the coordinate in each frame
    # NOTE. CONFIRMED. count would be the final number of total circles.

    tail_length = (tailfit[0][-1,:]-tailfit[0][0,:]).max()   #tag
    # Question. difficult why calculate tail_length in such way?

    return [(frame-frame[0,:]).astype(float)/float(tail_length) for frame in tailfit]
    # Question. why normal this way?
    # Question. whether this return could normalize to 0 to 1? doubt it.
    # ndarray.astype(dtype, order='K', casting='unsafe', subok=True, copy=True).  Copy of the array, cast to a specified type.


def sliding_average(somelist, window_size = 10):
    # var = sliding_average(numpy.abs(numpy.diff(sliding_average(bendangle,4))))

    somelistpadded = numpy.lib.pad(somelist,(window_size/2,window_size/2),'edge')   #tag
    # Note. numpy.pad(array, pad_width, mode, **kwargs), it Pads an array.
    # Note. 'edge', Pads with the edge values of array.
    # Question. does the window_size matters?
    # Question. why do the padding?
    # Question. why use somelistpadded as kernel? what is the output of the convolution?

    # Note. this convolution would add one more items to the returned list
    # Note. by sliding [0.25,0.25,0.25,0.25] through the angles list, the curve is smoothed by average effect
    # numpy.convolve(a, v, mode='full')[source], If v is longer than a, the arrays are swapped before computation.
    return numpy.convolve(somelistpadded, numpy.ones(int(window_size))/float(window_size),mode='valid')

def sliding_gauss(somelist, window_size = 10,sigma=3):
    # ultrasmoothvar = sliding_gauss(var,40,20)
    somelistpadded = numpy.lib.pad(somelist,(window_size/2,window_size/2),'edge')
    normpdf = scipy.stats.norm.pdf(range(-int(window_size/2),int(window_size/2)),0,sigma)
    return numpy.convolve(somelistpadded,  normpdf/numpy.sum(normpdf),mode='valid')


def extractbouts(bendangle, thresh=.35, min_bout_len = 10):
    # boutedges, var = extractbouts(angles, threshval)
    # Note. the function that extract each bouts from the video?
    # self.smooth = extractbouts(angles)[1]

    ''' DT-code
    plt.plot(numpy.asanyarray(bendangle),'r')
    plt.plot(sliding_average(bendangle,4),'b')
    plt.plot(numpy.diff(sliding_average(bendangle,4)),'y')
    plt.plot(numpy.abs(numpy.diff(sliding_average(bendangle,4))),'g')
    plt.plot(sliding_average(numpy.abs(numpy.diff(sliding_average(bendangle,4)))),'c')
    plt.show()
    '''

    var = sliding_average(numpy.abs(numpy.diff(sliding_average(bendangle,4))))
    # numpy.diff(a, n=1, axis=-1), Calculate the n-th order discrete difference along given axis.
    # Question. know the grammar know, but the real meaning?

    smoothvar = sliding_average(var, 4)
    ultrasmoothvar = sliding_gauss(var,40,20)
    # Note. gaussian kernel with a larger size just do a better job in smooth, leaving only the major peak without detailed osillation
    # Note. you can got similar result by enlarge the kernel in sliding_average

    ''' DT-code
    plt.plot(ultrasmoothvar+2,'b')
    plt.plot(smoothvar,'r')
    plt.show()
    '''

    bouts = ultrasmoothvar>thresh
    dbouts=numpy.diff(bouts)
    # Note. edge of bouts is in dbouts now!    #tag

    #if at edges are bout, add to boutedge
    boutedges=numpy.where(dbouts)[0]
    if bouts[0]:
        boutedges=numpy.insert(boutedges,0, 0)
    if bouts[-1]:
        boutedges=numpy.append(boutedges, len(bouts))
    assert len(boutedges)%2 ==0
    boutedges=[(boutedges[i],boutedges[i+1]) for i in range(0,len(boutedges),2) if abs(boutedges[i]-boutedges[i+1])> min_bout_len] #tuple-ize
    return boutedges, ultrasmoothvar

def boutacceptable(tailfit):
    # Note. seems to be a function to judge whether a bout is acceptable?

    normtailfit = normalizetailfit(tailfit)
    angles = tail2angles(normtailfit)
    nans = numpy.isnan(angles) + numpy.isinf(angles)
        
    if any(nans):
        return False

    if any([len(f)<25 for f in tailfit]):
        return False

    if abs(normtailfit[0][0,0] - min([f[:,0].min() for f in normtailfit]))>.05:
        return False
    
    return True


class boutplotter(object):
    # ibouts = boutplotter(angles, colors[behavior], threshinit, title=os.path.basename(video))

    def __init__(self,angles, color='b', threshinit = .45, title = None):
        '''PLOT THE BOUTS'''
        # with slider to change the frame
        # with interactive code to record key press etc.

        self.default_span_alpha = .4
        self.angles = angles
        self.color = color
        self.threshinit = threshinit
        
        self.figure = pylab.figure(figsize=[6*2,4*2])
        # Note. Create figure class. Figure([figsize, dpi, facecolor, edgecolor, ...])	The Figure instance supports callbacks through a callbacks attribute which is a matplotlib.cbook.CallbackRegistry instance

        pylab.subplots_adjust(bottom=.2)

        self.smooth = extractbouts(angles)[1]    #tag
        self.spanhandles = []
        self.hlinehandles = []

        pylab.subplot(211)
        pylab.plot(self.angles,self.color,)[0]
        pylab.subplot(212)
        pylab.plot(self.smooth,'k', figure=self.figure)
        # Note. plt.show() is neccessary before showing the plot result

        self.toggle_clicks = []

        if title:
            pylab.title(title)
        #self.title = title
        
        self.threshslider = Slider(pylab.axes([0.15, 0.1, 0.70, 0.03], axisbg='r'),'Thresh',0,self.smooth.max()/2,valinit=self.threshinit)
        self.threshslider.on_changed(self.update)

        self.update(self.threshinit)

        self.cids = []
        self.cids += [self.figure.canvas.mpl_connect('button_press_event', self.toggle_click)]
        self.cids += [self.figure.canvas.mpl_connect('key_press_event', self.quit_figure)]
        self.cids += [self.figure.canvas.mpl_connect('close_event', self.close)]
        # mpl_connect(s, func), Connect event with string s to func.
        
        self.show()   #tag


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
        self.spanhandles = []
        self.hlinehandles = []
        
    def update(self,val):
        boutedges,var = extractbouts(self.angles,val)
        self.boutedges = boutedges
        self.clear_handles()
        
        pylab.subplot(211)
        pylab.xlim((0,len(self.angles)))
        self.spanhandles = [pylab.axvspan(i[0],i[1],alpha = self.default_span_alpha,color=self.color,figure=self.figure) for i in boutedges]
        pylab.subplot(212)
        pylab.xlim((0,len(self.angles)))
        self.hlinehandles += [pylab.axhline(val,alpha=1,color=self.color,figure=self.figure)]
        
        #this redraws the previously clikced on spans
        clicks = list(self.toggle_clicks) #to ensure a deep copy
        self.toggle_clicks = []
        [self.toggle_click_point(point, True) for  point in clicks]
        
        pylab.draw()
        self.thresh = self.threshslider.val
        self.bouts = self.get_bouts()
        
    def show(self):
        pylab.show()
        return self.thresh

    def get_bouts(self):
        # return bout in list
        return [bout for axv, bout in zip(self.spanhandles, self.boutedges) if not any([axv.contains_point(click) for click in self.toggle_clicks])]
    
    def toggle_click(self, event):
        if event.button == 1:
            self.toggle_click_point((event.x, event.y))
            
    def toggle_click_point(self, point, skip_draw = False):
        # Note. difficult

        for axv in self.spanhandles:
            if axv.contains_point(point):   #tag
                
                previous_clicks = [axv.contains_point(click) for click in self.toggle_clicks]

                if any(previous_clicks):
                    self.toggle_clicks.pop(previous_clicks.index(1))
                    axv.set_alpha(self.default_span_alpha)
                    if not skip_draw:
                        pylab.draw()
                else:
                    axv.set_alpha(.1)
                    if not skip_draw:
                        pylab.draw()

                    extents = axv.get_extents() 
                    center = ((extents.x0+extents.x1)*.5,(extents.y0+extents.y1)*.5)  #use the span center instead of the point, incase the span gets shrunken
                    self.toggle_clicks.append(center)

        self.bouts = self.get_bouts()

def interactive_bouts(trial, *args, **kwargs):
    b = boutplotter(trial, *args, **kwargs)
    return b.bouts


def boutplothelper(metric, results, filename = None):
    bouts = [i['bout'] for i in results]
    pylab.plot(metric)
    if filename:
        pylab.title(filename)
    [pylab.axvspan(i[0],i[1],alpha=.3,color='r') for i in bouts]
    pylab.ylim([min([-60, min(metric)*1.25]), max([60, max(metric)*1.25])])
    [pylab.text(.5*(i['bout'][0] + i['bout'][1]), min(metric)*1.15,i['behavior'], horizontalalignment = 'center')  for i in results]
    pylab.show()

def boutplothelper2(metrics, results, filename = None):
    bouts = [i['bout'] for i in results]
    for metric in metrics:
        pylab.plot(metric)
    
    if filename:
        pylab.title(filename)
    [pylab.axvspan(i[0],i[1],alpha=.3,color='r') for i in bouts]
    pylab.ylim([min([-60, min(metric)*1.25]), max([60, max(metric)*1.25])])
    [pylab.text(.5*(i['bout'][0] + i['bout'][1]), min(metric)*1.15,i['behavior'], horizontalalignment = 'center')  for i in results]
    pylab.show()


def plottail(bout):
    [pylab.plot(b[:,0],b[:,1],color=pylab.cm.gist_rainbow(i/float(len(bout['tail'])))) for i,b in enumerate(bout['tail'])];pylab.show()


if __name__ == '__main__':
    fake = numpy.zeros(2000)
    fake[400:1100] = numpy.sin(numpy.linspace(0,90,1100-400))*20*numpy.sin(numpy.linspace(0,4,1100-400))
    fake[1500:1800] = numpy.sin(numpy.linspace(0,40,300))*10
    b=boutplotter(fake)
    print b.bouts
    print "done"


