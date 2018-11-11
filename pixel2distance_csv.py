import csv
import os
from matplotlib import pyplot as plt
import pandas as pd



def dt_csv_to_list(filename, distance_list, size_list):
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        list = [row for row in reader]
        for i in range(len(list)-1):
            distance_list.append(list[i+1][3])
            size_sb_list.append(list[i+1][4])
    return None


if __name__ == "__main__":


    # input here
    ratio = 0.011363
    folder = 'G:\\DT\\2018\\Jan\\Jan 12th\\results'   # this should be the folder where you put all the videos for analysis
    output_path = folder

    # search all the csvs in that folder
    filenames = os.listdir(folder)
    csv_list = [filename for filename in filenames if os.path.splitext(filename)[1] == '.csv']

    # start analysis one by one
    for csv_file in csv_list:

        #read each csv first, extract bladder distance and size, store them in new list
        bladder_mid_list = []
        size_sb_list = []
        dict = dt_csv_to_list(folder+'\\'+csv_file, bladder_mid_list, size_sb_list)

        #calculate the ratio for sizes
        size_ratio = ratio*ratio # mm2 for each pixel

        # pixel2distance
        for i in range(len(bladder_mid_list)):
            bladder_mid_list[i] = float(bladder_mid_list[i])*ratio
            #size_sb_list[i] = float(size_sb_list[i])*size_ratio

        data_dict = {'bladder_distance':bladder_mid_list, 'bladder_size':size_sb_list}
        df = pd.DataFrame(data_dict)

        ### SAVE RESULTS ###
        df.to_csv(output_path + '\\' + 'distance_mm_' + csv_file )

        ''' Plotting part
        # normalize the size_sb_list
        n_size_sb_list = []  # normalized the size of swimbladder before plotting
        for item in size_sb_list:
            n_size_sb_list.append(item / max(size_sb_list) * min(bladder_mid_list))

        # plot and automatically save them
        plt.plot(bladder_mid_list, 'b')
        plt.plot(n_size_sb_list, 'g')
        '''





