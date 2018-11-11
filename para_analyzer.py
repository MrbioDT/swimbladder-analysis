import csv
import os

def letter2list(input_list):
    #convert a list contains a lot of letter into a list with numbers
    tem_list = []
    tem_str = ''
    for letter in input_list:
        try:
            number = int(letter)
            tem_str += letter
        except:
            if len(tem_str) != 0 and letter != '.' and len(tem_str)<5: #use a series of condition to make the right judgement about number
                tem_list.append(int(tem_str))
                tem_str = ''
            if letter == ')':
                tem_str = ''
    return tem_list

# READ ALL THE AVIS FILES WITHIN THAT FOLDER
folder = 'G:\DT-data\\2018\May\May 28_low density\\20180528_2nd\\results\para'
filenames = os.listdir(folder)
csv_list = [filename for filename in filenames if os.path.splitext(filename)[1] == '.csv']
result_list = []


for csv_file in csv_list:

    csv_name = folder + '\\' + str(csv_file)
    csvFile = open(csv_name, "r")
    reader = csv.reader(csvFile)
    # reader is:  <_csv.reader object at 0x04BB0470>
    # the type of reader is:  <type '_csv.reader'>

    result = {}
    headers = next(reader)
    contents = next(reader)

    for index, item in enumerate(headers):
        if index != 0:
            result[item] = contents[index]

    result_list.append(result)
    csvFile.close()

annotation_list = []
strike_list = []

count = 1
for csv_dict in result_list:

    print 'this is the ', count, 'th one, original striking is: ', csv_dict['final_striking']
    count += 1

    tem_annotation_list = letter2list(csv_dict['annotation'])
    tem_striking_list = letter2list(csv_dict['final_striking'])
    annotation_list.append(tem_annotation_list)
    strike_list.append(tem_striking_list)

print 'strike_list is: ', strike_list
print 'annotation_list is: ', annotation_list

matched = 0
false_positive = 0
false_negative = 0

for i in range(len(annotation_list)):
    for s_item in strike_list[i]:
        if len(annotation_list[i]) == 0:
            #situation no.1 if the annotation_list is empty
            false_positive += len(strike_list[i])
            print '[FALSE POSITIVE] strike ', strike_list[i], ' is false positive in the file ', csv_list[i]
            break
            # break! otherwise the same list will be re-counted!
        else:
            #situation no.2 if the annotation list is not empty!
            counting = 0
            for a_item in annotation_list[i]:
                counting += 1
                if abs(a_item - s_item) <= 20:
                    break
                else:
                    if counting == len(annotation_list[i]) and len(annotation_list[i]) != 0:
                        false_positive += 1
                        print '[FALSE POSITIVE] strike ', s_item, ' is false positive in the file ', csv_list[i]

    for a_item in annotation_list[i]:
        if len(strike_list[i]) == 0:
            #situation no.1 if the strike_list is empty
            false_negative += len(annotation_list[i])
            print '[FALSE NEGATIVE] annotation ', annotation_list[i], ' is false negative in the file ', csv_list[i]
            break
            # break! otherwise the same list will be re-counted!
        else:
            #situation no.2 if the strike list is not empty!
            counting = 0
            for s_item in strike_list[i]:
                counting += 1
                if abs(a_item - s_item) <= 20:
                    matched += 1
                    print '[MATCHED] annotation ', a_item, ' is matched in the file ', csv_list[i]
                    break
                else:
                    if counting == len(strike_list[i]) and len(strike_list[i]) != 0:
                        false_negative += 1
                        print '[FALSE NEGATIVE] annotation ', a_item, ' is false negative in the file ', csv_list[i]


# for i in range(len(strike_list)):
#     for s_item in strike_list[i]:
#         if len(annotation_list[i]) == 0:
#             print 'enter when the s_item is: ', s_item
#             false_positive += len(strike_list[i])
#         else:
#             counting = 0
#             for a_item in annotation_list[i]:
#                 counting += 1
#                 if abs(a_item - s_item) <= 10:
#                     break
#                 else:
#                     if counting == zelen(annotation_list[i]):
#                         print counting, len(strike_list[i])
#                         false_positive += 1

print 'matched, false_positive, false_negative are : ', matched, false_positive, false_negative
print 'accuracy in terms of false positive rate is: ', matched/float(matched+false_positive)
print 'accuracy in terms of false negative rate is: ', matched/float(matched+false_negative)












