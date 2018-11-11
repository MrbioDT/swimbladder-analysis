# THIS CODE WILL READ CERTAIN COLUMN OF CSV FILES
# refering this link: http://python3-cookbook.readthedocs.io/zh_CN/latest/c06/p01_read_write_csv_data.html
#
# ---------------- USER INPUTS ---------------------
#
# file = '4th_06_s__12_19_11_56_7_310_thresh_207_mthresh_74_mroi_((266, 304), (287, 323))_realtime_version.csv' # the name and location of the target csv files
# column_index = 5 # the index of the column you want to read
#
# ----------------OUTPUTS------------------------
#
# data_list, the column you want to read
# plotting of the data_list, optional
#
# -------------------------------------------------
# Last update: 07 MAY 2018, DT

import csv
from collections import namedtuple
from matplotlib import pyplot as plt


if __name__ == '__main__':

    file = '4th_06_s__12_19_11_56_7_310_thresh_207_mthresh_74_mroi_((266, 304), (287, 323))_realtime_version.csv'
    column_index = 5

    data_list = []
    with open(file) as f:
        f_csv = csv.reader(f)
        headings = next(f_csv)  # read the headings of csv
        for r in f_csv:  # read each row one by one
            data_list.append(float(r[column_index]))
        plt.plot(data_list)
        plt.show()
    print data_list



