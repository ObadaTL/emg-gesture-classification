
from scipy.io import loadmat
from pathlib import Path
from glob import glob
from openpyxl import Workbook
import pandas as pd
import numpy as np
import csv

class matImporter():
    def __init__(self, dataDirectory, listOfFiles):
        for file in listOfFiles:
            filePath = str(dataDirectory / file)
            fileData = loadmat(filePath)
            pass

class UCIImporter():
    def __init__(self, dataDirectory):
        self.dir = dataDirectory

    def toXLSX(self):
        outXLSX = Workbook()
        del outXLSX['Sheet']
        for i in range(1,37):

            idxStr = str(i)
            if i < 10:
                idxStr = '0{}'.format(idxStr)
            subDir = self.dir / idxStr
            for file in glob('{}/*.txt'.format(str(subDir))):
                print(file)
                filePath = Path(str(self.dir / file))
                ws = outXLSX.create_sheet(filePath.stem)

                with open(file) as inFile:

                    ws.append(['time', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'id', 'class'])
                    inFile.readline()

                    for line in inFile:
                        data = line.strip('\n').split('\t')
                        data.insert(-1, i)
                        ws.append(data)

            outXLSX.save(filename = '{}/UCI.xlsx'.format(self.dir))

    def toCSV(self):

        header = (['time', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'id', 'idx', 'class'])
        for i in range(1,37):
            idxStr = str(i)
            if i < 10:
                idxStr = '0{}'.format(idxStr)
            subDir = self.dir / idxStr
            count = 0

            idOutFile = subDir / '{}.csv'.format(idxStr)

            with idOutFile.open('w', newline='') as id_file:
                writer = csv.writer(id_file)
                writer.writerow(header)

                for file in glob('{}/*.txt'.format(str(subDir))):
                    print('\t{}'.format(file))
                    count += 1
                    outFile = subDir / '{}-{}.csv'.format(idxStr, count)

                    with outFile.open('w', newline='') as out_file:
                        writer2 = csv.writer(out_file)
                        writer2.writerow(header)

                        with open(file) as f:
                            f.readline()
                            for line in f:
                                data = line.strip('\n').split('\t')
                                data.insert(-1, count)
                                data.insert(-2, i)
                                writer.writerow(data)
                                writer2.writerow(data)
            print(idOutFile)


class CSVHandler():
    def __init__(self, config):
        self.users = []

        for user in range(1, config.users+1):

            if user < 10:
                usrIdx = '0{}'.format(user)
            else:
                usrIdx = str(user)

            filePath = Path(config.data).joinpath(usrIdx).joinpath('{}.csv'.format(usrIdx))
            self.users.append(pd.read_csv(filePath))


        self.users[0].head()
        emg_data = self.users[0].copy()

        titles = ['time', 'channel1', 'channel2', 'channel3', 'channel4', 'channel5'
                                                                          'channel6', 'channel7', 'channel8', 'class']

        # Remove Columns
        time = emg_data.pop('time')
        # emg_data.pop('channel5')
        # emg_data.pop('channel6')
        # emg_data.pop('channel7')
        # emg_data.pop('channel8')
        emg_data = np.array(emg_data)

        # The matrix of the data collected (segment dimension, channel no)
        row = len(emg_data)
        col = len(emg_data[0])
        # print("TYPE", type(emg_data))
        print(f"The size of the emg data is {row} x {col}")
        # print(emg_data)

        emg_X, emg_y = emg_data[:, :-1], emg_data[:, -1]
        no_channels = len(emg_X[0])
        print(emg_X.shape, emg_y.shape)
        # feature_no = len(X[0])
        # print(feature_no)
        print(emg_X)


class txtImporter():
    def __init__(self, dataDirectory, listOfFiles):
        for file in listOfFiles:
            filePath = str(dataDirectory / file)
            fileData = loadmat(filePath)
            pass

# class DataHandler():
#     def __init__(self, path):
# listofFiles = ['recording{}'.format(str(i)) for i in range(1,21)]
# fileConvertor = matImporter(Path('./Data'), listofFiles)

# importer = UCIImporter(Path('./Data/UCI'))
# importer.toCSV()