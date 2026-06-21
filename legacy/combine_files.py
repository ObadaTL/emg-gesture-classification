import pandas
import glob
import csv

#Collects the files names of all .txt files in a given directory.
file_names = glob.glob("/Users/rubytan/Downloads/EMG_data_for_gestures-master/combine4/*")
csv_file = '/Users/rubytan/Desktop/emg_data/combine_4/1.csv'
txt_file = '/Users/rubytan/Desktop/emg_data/combine_4/1.txt'

header_list = ['time', 'channel1', 'channel2', 'channel3', 'channel4', 'channel5',
               'channel6', 'channel7', 'channel8', 'class']
#[Middle Step] Merges the text files into a single file titled 'output_file'.
with open(txt_file, 'w') as out_file:
    out_file.write('time'+ "\t" + 'channel1'+ "\t" + 'channel2'+ "\t" + 'channel3'
                   + "\t" + 'channel4'+ "\t" + 'channel5'+ "\t" + 
                   'channel6'+ "\t" + 'channel7'+ "\t" + 'channel8'+ "\t" + 'class' + "\n")
    for path in file_names:
        with open(path) as in_file:
            next(in_file)
            for j in in_file:
                out_file.write(j)

#Reading the merged file and creating dataframe.
data = pandas.read_csv(txt_file, delimiter = '\t')
print(data)
  
#Store dataframe into csv file.
data.to_csv(csv_file, index = None)


# In[ ]:




