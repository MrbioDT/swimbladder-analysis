from scipy import signal
import numpy as np
from matplotlib import pyplot as plt
import peakdetector
import matplotlib.pyplot as plt
from openpyxl import load_workbook
import os


def tail_analysis(striking_candidates, freq, tail_thresh = 50):
    # this is the function to filtered out striking candidates with large tail bend amplitude
    # usage: striking_candidates = tail_analysis(striking_candidates, freq, tail_thresh = 50)
    # add another function to judge if tail movement is uni-directional or bi-directional

    new_striking_candidates = []

    for frame in striking_candidates:
        inbout, bout, bout_index = frame_in_bouts(frame[0],freq[3])
        # freq[3] contains the framerange of each bouts, it should be a list with tuple as each item, each tuple have two value
        # inbout is a boolean refects if the frame is in the bout
        if inbout:
            max_amplitude = max(freq[1][bout_index])
            min_amplitude = min(freq[1][bout_index])
            print max_amplitude
            print min_amplitude
            # first, check out if the tail movement is bi-directional
            # 10 degree here is the threshold for judging the tail movement
            if abs(max_amplitude) < 15 or abs(min_amplitude) < 15:
                if abs(max_amplitude) < 65 and abs(min_amplitude) < 65:
                   print 'the tail bending amplitude is uni-direction in  ', str(frame), ' frame, which is ', max_amplitude, ' and ', min_amplitude
                   new_striking_candidates.append(frame)
            else:
                # secondly, check if the tail movement amplitude is bigger than 50 degree
                max_abs_amplitude = max(abs(freq[1][bout_index][:frame[0]]))
                if max_abs_amplitude < tail_thresh:
                   new_striking_candidates.append(frame)
                else:
                   print 'the tail bending amplitude is too big in ', str(frame), ', which is ', max_abs_amplitude, ' degree!'
        else:
            print 'ops...', str(frame), ' is not in the bout so it does not count'

    return new_striking_candidates

def frame_in_bouts(frame,boutedges):
    # this is the function to judge if the frame is within certain bout, if so return true and corresponding bouts
    # boutedges should be a list with tuple as each item, each tuple have two value
    for i, each_tuple in enumerate(boutedges):
        if each_tuple[0] < frame < each_tuple[1]:
            return True, each_tuple, i
    return False, (0,0), -1


# # Some notes about freq
# # freq[0][n] contains the frequencies of the nth bout
# # freq[1][n] contains the angles of the nth bout
# # freq[2][n] contains the x,y coordinates of each peak of the nth bout
# # freq[3][n] contains the framerange of each bouts
# # freq[4][n] contains the name of the nth video within that shvs (shvname_list)

def get_annotation(folder):
    # input should be the folder that contains the annotation xlsx files

    filenames = os.listdir(folder)
    xlsx_list = [filename for filename in filenames if os.path.splitext(filename)[1] == '.xlsx']
    xlsx_list = [xlsx_list[0]] # just take the first item

    if len(xlsx_list) == 1:
        xlsx_path = folder + '\\' + xlsx_list[0]
        ### Create a annotation list based on the annotation xlsx
        wb = load_workbook(xlsx_path)
        sheet_ranges = wb['Sheet1']
        annotation_list = []
        for i, item in enumerate(sheet_ranges['k']):  # following B/K can be adjust to read different column
            if sheet_ranges['k'][i].value != None:
                annotation_list.append([sheet_ranges['B'][i].value, sheet_ranges['k'][i].value])

        ### Create a true list based on annotation list
        true_list = []

        for i, item in enumerate(annotation_list):
            if str(item[1]) != 'none' and str(item[1]) != 'strike frames':
                print 'this is, ', item
                tem_list = []
                tem_str = ''
                tem = tuple(str(item[1]))

                for x, letter in enumerate(tem):
                    try:
                        number = int(letter)
                        tem_str += letter
                    except:
                        if tem_str != '':
                            tem_list.append(int(tem_str))
                            tem_str = ''
                    if tem_str != '' and x == len(tem) - 1:
                        tem_list.append(int(tem_str))
                        tem_str = ''

                if tem_list != []:
                    true_list.append((str(item[0]), tem_list))

        print 'true list for ', xlsx_list, ' is: ', true_list
        return true_list



def plot_strike_candidates(binocular, sb_mid, sb_cr, tail_amplitude_list, striking_candidates, mouth_area = None, video = None, mouth_ana = '', annotation_list = []):

    plt.figure(figsize=(20, 10)) #try to display a big figure

    if mouth_ana == 'y':
        plt.subplot(511)  # the first subplot in the first figure
        plt.plot(binocular)

        threshold_line = []
        for i in binocular:
            threshold_line.append(30) #set the threshold line for eye movement
        plt.plot(threshold_line,'g')
        plt.ylim(0,max(binocular)+10)   # set the y-range for the ploting
        plt.xlim(0,len(sb_mid)+10)   # set the x-range for the ploting

        plt.title('Binocular convergence')
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        if annotation_list != []:
            for i in annotation_list:
                plt.scatter(i, binocular[i], marker='v', c='g', edgecolors='face', s=50)
        if striking_candidates!= []:
            for frame in np.array(striking_candidates)[:, 0]:
                plt.scatter(frame,binocular[frame],marker='^',c='r',edgecolors='face', s=50)
                # y = np.arange(min(binocular), max(binocular), 0.001)
                # x = []
                # for i in y:
                #     x.append(frame)
                # plt.plot(x, y, 'r')

        plt.subplot(512)  # the second subplot in the first figure
        plt.plot(sb_mid)
        plt.xlim(0,len(sb_mid)+10)   # set the x-range for the ploting
        plt.title('swim bladder actual moving distance')
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        if annotation_list != []:
            for i in annotation_list:
                plt.scatter(i, sb_mid[i], marker='v', c='g', edgecolors='face', s=50)
        if striking_candidates != []:
            for frame in np.array(striking_candidates)[:, 0]:
                plt.scatter(frame,sb_mid[frame],marker='^',c='r',edgecolors='face', s=50)
                # y = np.arange(min(binocular), max(binocular), 0.001)
                # x = []
                # for i in y:
                #     x.append(frame)
                # plt.plot(x, y, 'r')


        plt.subplot(513)  # the third subplot in the first figure
        plt.plot(sb_cr)
        plt.xlim(0,len(sb_mid)+10)   # set the x-range for the ploting

        plt.title('changing rate of swim bladder movement')
        if annotation_list != []:
            for i in annotation_list:
                plt.scatter(i, sb_cr[i], marker='v', c='g', edgecolors='face', s=50)
        if striking_candidates != []:
            for frame in np.array(striking_candidates)[:, 0]:
                plt.scatter(frame,sb_cr[frame],marker='^',c='r',edgecolors='face', s=50)
                # y = np.arange(min(binocular), max(binocular), 0.001)
                # x = []
                # for i in y:
                #     x.append(frame)
                # plt.plot(x, y, 'r')

        plt.subplot(514)  # the third subplot in the first figure
        plt.plot(mouth_area)
        plt.xlim(0,len(sb_mid)+10)   # set the x-range for the ploting

        plt.title('mouth open (area)')
        if annotation_list != []:
            for i in annotation_list:
                plt.scatter(i, mouth_area[i], marker='v', c='g', edgecolors='face', s=50)
        if striking_candidates != []:
            for frame in np.array(striking_candidates)[:, 0]:
                plt.scatter(frame,mouth_area[frame],marker='^',c='r',edgecolors='face', s=50)
                # y = np.arange(min(binocular), max(binocular), 0.001)
                # x = []
                # for i in y:
                #     x.append(frame)
                # plt.plot(x, y, 'r')

        plt.subplot(515)  # the third subplot in the first figure
        plt.plot(tail_amplitude_list)
        tail_threshold_line = []
        for i in tail_amplitude_list:
            tail_threshold_line.append(50) #set the threshold line for tail_movement
        plt.plot(tail_threshold_line,'g')
        plt.xlim(0, len(tail_amplitude_list) + 10)  # set the x-range for the ploting

        plt.title('tail_bend_amplitude(absolute)')
        if annotation_list != []:
            for i in annotation_list:
                plt.scatter(i, tail_amplitude_list[i], marker='v', c='g', edgecolors='face', s=50)
        if striking_candidates != []:
            for frame in np.array(striking_candidates)[:, 0]:
                frame = int(frame) #make the float a int
                plt.scatter(frame, tail_amplitude_list[frame], marker='^', c='r', edgecolors='face', s=50)
                # y = np.arange(min(binocular), max(binocular), 0.001)
                # x = []
                # for i in y:
                #     x.append(frame)
                # plt.plot(x, y, 'r')
    else:
        plt.subplot(411)  # the first subplot in the first figure
        plt.plot(binocular)

        threshold_line = []
        for i in binocular:
            threshold_line.append(30) #set the threshold line for eye movement
        plt.plot(threshold_line,'g')
        plt.ylim(0,max(binocular)+10)   # set the y-range for the ploting
        plt.xlim(0,len(sb_mid)+10)   # set the x-range for the ploting

        plt.title('Binocular convergence')
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        if striking_candidates!= []:
            for frame in np.array(striking_candidates)[:, 0]:
                plt.scatter(frame,binocular[frame],marker='^',c='r',edgecolors='face')
                # y = np.arange(min(binocular), max(binocular), 0.001)
                # x = []
                # for i in y:
                #     x.append(frame)
                # plt.plot(x, y, 'r')

        plt.subplot(412)  # the second subplot in the first figure
        plt.plot(sb_mid)
        plt.xlim(0,len(sb_mid)+10)   # set the x-range for the ploting
        plt.title('swim bladder actual moving distance')
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        if striking_candidates!= []:
            for frame in np.array(striking_candidates)[:, 0]:
                plt.scatter(frame,sb_mid[frame],marker='^',c='r',edgecolors='face')
                # y = np.arange(min(binocular), max(binocular), 0.001)
                # x = []
                # for i in y:
                #     x.append(frame)
                # plt.plot(x, y, 'r')

        plt.subplot(413)  # the third subplot in the first figure
        plt.plot(sb_cr)
        plt.xlim(0,len(sb_mid)+10)   # set the x-range for the ploting
        plt.title('changing rate of swim bladder movement')
        if striking_candidates!= []:
            for frame in np.array(striking_candidates)[:, 0]:
                plt.scatter(frame,sb_cr[frame],marker='^',c='r',edgecolors='face')
                # y = np.arange(min(binocular), max(binocular), 0.001)
                # x = []
                # for i in y:
                #     x.append(frame)
                # plt.plot(x, y, 'r')

        plt.subplot(414)  # the third subplot in the first figure
        plt.plot(tail_amplitude_list)
        plt.plot(50,'g') #50 degree here is the amplitude threshold
        plt.xlim(0, len(tail_amplitude_list) + 10)  # set the x-range for the ploting

        plt.title('tail_bend_amplitude(absolute)')
        if annotation_list != []:
            for i in annotation_list:
                plt.scatter(i, tail_amplitude_list[i], marker='v', c='g', edgecolors='face', s=50)

        if striking_candidates != []:
            for frame in np.array(striking_candidates)[:, 0]:
                frame = int(frame)
                plt.scatter(frame, tail_amplitude_list[frame], marker='^', c='r', edgecolors='face', s=50)
                # y = np.arange(min(binocular), max(binocular), 0.001)
                # x = []
                # for i in y:
                #     x.append(frame)
                # plt.plot(x, y, 'r')

    plt.tight_layout()
    print 'video.name is: ', video.name
    plt.savefig(video.name +  '_strike_detection.png')
    plt.clf()
    #plt.show()




def striking_detection(data, binocular, peak_thresh=0.03, temporal_thresh=15, amplitude_thresh=0.2):
    # input data has to be an iterate, either list or tuple
    # peak_thresh is the threshold for detecting peaks in function peakdetector.peakdetold(data,0.15)
    # temporal_thresh specifies the positive peak has to be within 'temporal_thresh' frames right after negative peaks
    # amplitude_thresh specifies the value for positive peak has to be 'amplitude_thresh' bigger than the negative peaks

    peakind = peakdetector.peakdetold(data,peak_thresh)
    # get the number of peaks which tells us about how many tail beats for the bout
    # the second number is the threshold, it compares the peaks with neighbor value

    p_p = peakind[0] #positive peaks
    n_p = peakind[1] #negative peaks
    striking_frame = []

    for i, p in enumerate(n_p):
        for sub_i, sub_p in enumerate(p_p):
            if 0 < p_p[sub_i][0] - n_p[i][0] < temporal_thresh:  # temporal threshold! #the positive peak has to be within 'temporal_thresh' frames right after negative peaks
               if p_p[sub_i][1] - n_p[i][1] > amplitude_thresh:  # peak amplitude threshold! #the value for positive peak has to be 'amplitude_thresh' bigger than the negative peaks
                  mid = (p_p[sub_i][0] + n_p[i][0]) / 2
                  if binocular[mid]>30:  #to make sure the binocular eyes convergence is bigger than 30
                     striking_frame.append((mid, data[mid]))
                     break # just break the inner for loop and go back to the outer for loop

    # plotting
    # plt.plot(data)
    # plt.scatter(np.array(peakind[0])[:, 0],np.array(peakind[0])[:, 1])  # peakind[0] give you the peaks for the positive peaks
    # plt.scatter(np.array(peakind[1])[:, 0], np.array(peakind[1])[:, 1], c='r',edgecolors='face')  # peakind[1] give you the peaks for the positive peaks
    # plt.show()

    #detect the the same strike predicted as the 2 frame
    orginal_striking_frame = striking_frame #save the frame for future reference

    if len(striking_frame) > 1:
        for i in range(0, len(striking_frame) - 1):
            if i >= len(striking_frame)-1:
                break
            if striking_frame[i + 1][0] - striking_frame[i][0] < 15:  # two strikes can not occur within an interval less than 30 frame during 300Hz sampling
                mid_x = (striking_frame[i + 1][0] + striking_frame[i][0]) / 2
                mid_y = data[mid_x]
                striking_frame[i] = (mid_x, mid_y)
                striking_frame.remove(striking_frame[i + 1])

    #just do twice to reduce extra close one...
    if len(striking_frame) > 1:
                for i in range(0, len(striking_frame) - 1):
                    if i >= len(striking_frame) - 1:
                        break
                    if striking_frame[i + 1][0] - striking_frame[i][0] < 15:  # two strikes can not occur within an interval less than 30 frame during 300Hz sampling
                        mid_x = (striking_frame[i + 1][0] + striking_frame[i][0]) / 2
                        mid_y = data[mid_x]
                        striking_frame[i] = (mid_x, mid_y)
                        striking_frame.remove(striking_frame[i + 1])

    # using the following code to give a line plotted on the corresponding striking frame
    if striking_frame != []:
        for frame in np.array(striking_frame)[:, 0]:
            y = np.arange(min(data), max(data), 0.001)
            x = []
            for i in y:
                x.append(frame)
        #     plt.plot(x, y, 'r')
        # plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)  # adjust the layout
        # print np.array(peakind[1][:,0])

    return striking_frame, orginal_striking_frame

if __name__ == "__main__":
    # this would be my original input
    sb_cr = [-0.00704155549339, 0.0195242185447, 0.0480643049212, 0.0287594283235, 0.0649069659932, -0.020196022178,
             -0.0146290515806, -0.00813542090292, -0.0505527064101, 0.00566861696018, -0.029641058888, -0.0253644895285,
             -0.00238770538856, -0.0345830954259, 0.00974176103856, -0.00772106624389, -0.00619830756668,
             0.0168727409559, -0.00894871193692, 0.0136364967436, 0.00186668883265, 0.0192935211008, 0.0205891048441,
             0.028420209439, 0.0312633817714, 0.00118612672378, -0.0102805671784, -0.0515740177283, -0.0176403346023,
             -0.0040772919629, -0.000797005534538, -0.00871874881225, -0.000915844289754, -0.0240786250474,
             -0.0245570681913, 0.0213973687456, 0.000980401278106, 0.029093197258, 0.00770063258534, 0.0278972031139,
             0.00305242020465, 0.0101135079178, -0.000244148789399, -0.0312488086478, -0.0120966505033,
             -0.0326127724866, 0.00767427605866, 0.00276865129194, 0.00645505932214, 0.0150033505141, -0.00962649878249,
             -0.00433716232327, 0.000327572921722, -0.000256487500641, 0.0280719769939, -0.0483418096803,
             -0.117096461443, -0.185784574135, -0.273523584444, -0.194714975046, -0.147088328594, -0.0480968035471,
             0.0438957576922, 0.0631542800908, 0.065861754858, 0.0550006255165, 0.0443473233967, 0.0422269300614,
             0.0707629253379, 0.0349360146533, 0.0312657614548, 0.0182331301955, -0.00725294987518, -0.00172990604882,
             0.0239115617242, 0.0157624199117, -0.000999427957198, 0.0313920380985, -0.00207470821338,
             -0.000459057038329, 0.00138648839609, 0.00586893639749, 0.0405318935853, 0.0349134522099, 0.045565824205,
             0.00131892235435, -0.0133167408437, -0.0132947815771, -0.0302049986018, -0.0793866667485, -0.153233910878,
             -0.200758201212, -0.191160890936, -0.142340306303, -0.0835350967171, -0.00350185486692, 0.034899753888,
             0.0709895931296, 0.0875814392053, 0.0664558721658, 0.0402206214175, 0.0319534229538, 0.0259075060432,
             0.0263048280274, 0.0885569701812, 0.0678318723647, 0.0422393331548, 0.0346230020091, -0.0177109164764,
             -0.00776167629265, 0.010618954841, 0.0268090470159, 0.0148544032746, 0.0193762298696, 0.0271607745882,
             0.0315047103225, 0.0207570635858, 0.0111639099467, -0.00406971699525, -0.0201067001175, 0.00193837023147,
             0.0190062301721, 0.0311988837614, 0.0272191055045, -0.00758889146902, 0.00814917767216, 0.00109002957541,
             0.000150685997598, 0.0439413920735, 0.00724520481561, 0.0118884860627, 0.0148254979288, -0.00554155249898,
             0.0243940765849, 0.0347514158826, -0.0041697316401, -0.00683280188759, -0.0108365468507, -0.0315724477759,
             0.0188347426502, -0.00551312995775, 0.00622289518937, 0.00892754293568, -0.0133953778225, 0.0325574592069,
             0.0371925248202, 0.0157467484326, 0.0116085531033, -0.0110393515563, -0.0247692130058]

