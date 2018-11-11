from injectargs import *
import pdb

class tailfitresult(object):
    # Question. why so many class inherent from object
    @injectArguments # Question. meaning of @?
    def __init__(self,tailfit, filename, path, startpoint,FPS, numframes, direction, shahash, tailfitversion):
        pass

    def __str__(self,):
##        
        strings = self.__dict__.keys()
        strings.pop(strings.index('tailfit'))
        
##        values = [ for string in strings]
        strings = [string +': '+str(self.__dict__[string])+'  ' for string in strings]
        strings.insert(0,"Tailfit from video: "+str(self.filename)+' with length ' +str(len(self.tailfit)) +'\n')
        return ''.join(strings)
        
        
