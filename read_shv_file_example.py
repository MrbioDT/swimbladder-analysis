import shelve
from filepicker import *


shv_path = pickfile()

shv = shelve.open(shv_path)
print shv.keys()
print shv[shv.keys()[0]]
print shv[shv.keys()[0]].FPS
shv.close()
