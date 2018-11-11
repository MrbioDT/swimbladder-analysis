# DT-tools
# Take all the files in the folder with '.avi' in filename, and put them input a csv file

import csv
import os

folder = 'D:\\DT\\2017\\December\\Dec 19th'
filenames = os.listdir(folder)
avis = [filename for filename in filenames if os.path.splitext(filename)[1] == '.avi']
avis = sorted(avis)

### CREATE A FOLDER WHERE RESULTS WILL BE SAVED ###
output_folder = os.path.join(folder)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


ofile  = open(output_folder+'\\avis.csv', "wb")
writer = csv.writer(ofile)

writer.writerow(avis)

ofile.close()