import numpy
import math
import pdb
import matplotlib.pyplot as plt


def tailfraction(fraction, fit):
    # Note. tailfraction(1-fraction,fit)
    # Question. what is the function of the tailfraction?
    assert fraction >= 0 and fraction <= 1
    return int(fraction*fit.shape[0])  #tag

def tail2mean (tailfit, fraction = .2):   #DT-TAG
    results=numpy.empty(len(tailfit))
    for i,fit in enumerate(tailfit):
        results[i]=fit[tailfraction(fraction,fit):,0].mean()
    return results   

def tail2angles(tailfit, fraction = .2):
    # Note. STAR. PARAMETERS. fraction here controls the tail_angle!!!
    # Note. Calculate the tail_angle!
    # Note. this function takes tailfit result, for each frame calculate the vector from the startpoint of fitting to the mean point of fraction of tail end
    # Note. extract the angles of vectors and store the angle of each frame in the returned list.
    # angles = tail2angles(tailfit)
    results = numpy.empty(len(tailfit))
    # Note. numpy.empty, Return a new array of given shape and type, without initializing entries.

    for i,fit in enumerate(tailfit):
        # in tailfit, each fit should corresponds to the fitting in one frame
        temp = fit[0,:]-numpy.mean(fit[tailfraction(1-fraction,fit):,:],0)  #tag
        # numpy.mean(fit[tailfraction(1-fraction,fit):,:],0) takes average point of last 20% of the tail and use to construct the vectors from the start point
        # fraction is given in the argument, equals to .2
        # tailfraction returns int(0.8*tail_length)
        # Note. temp is a vector from the startpoint of fitting to the mean point of fraction of tail end
        # Note. fit[0,:] equals to fit[0], access to the nth points in fitting

        results[i] = math.degrees(numpy.arctan2(temp[0],-temp[1])) # using this line when tail points down
                                                                   # simple correction by DT for down side tail movement

        # results[i] = math.degrees(numpy.arctan2(temp[0],temp[1]))+90 # using this line when tail points right
        # Note. add 90 degree, is rotated the vectors anticlockwise 90 degree
    return results


def tail2tipangles(tailfit):
    results=numpy.empty(len(tailfit))
    for i,fit in enumerate(tailfit):
        temp= fit[tailfraction(.8,fit),:]-numpy.mean(fit[-3:-1,:],0)
        results[i]=math.degrees(numpy.arctan2(temp[1],-temp[0]))
    return results


def tail2sumangles(tailfit):
    # Note. this function calculate the vector degree between each fitted points and return the mean value of all the angles in each frame
    # Note. returned results is a list with the total items corresponds to each frame, items value is the mean of tail angle

    # Note. tailfit correspond to fitted_tail, which is the following:
    # NOTE. CONFIRMED. fitted_tail is the final result of the whole program.
    # NOTE. CONFIRMED. it is a list, storing arrays with the total number of total frame analyzed/read
    # NOTE. CONFIRMED. each array corresponds to one frame, storing x number of points (x=tail_lengh/count)
    # NOTE. CONFIRMED. points is the fitted result_point(mid point of tail edge), it is the coordinate in each frame
    # NOTE. CONFIRMED. count would be the final number of total circles.

    results=numpy.empty(len(tailfit))

    for i,fit in enumerate(tailfit):   ############tag###########
        temp = fit[1:,:]-fit[:-1,:]
        # Note. -1 is the second last item
        results[i]=numpy.mean(abs(numpy.degrees(numpy.arctan2(temp[:,1],temp[:,0]))))
        # Question. what is the actual meaning of this results[i]?
        # Answer. I think it reflects that: to what extent is the tail bended. (average value)
    return results


def tail2mean2 (tailfit):
    results=numpy.array([])
    for fit in tailfit:
        results=numpy.append(results,fit[10:,1].mean())
    return results       

