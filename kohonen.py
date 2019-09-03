'''
Student Name: Nur Muhammad Bin Khameed
SRN: 160269044
CO3311 Neural Networks CW1

Main Class

Instructions:
Please install these libraries to run this file:
os, sys, xlrd, openpyxl, numpy, pandas, matplotlib
'''

import os.path
import sys
import xlrd
import openpyxl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Network:
    def __init__(self, no_of_units, learning_rate, terminating_weight_change, dataset, classes=None, epochs_to_run=0):

        self.dataset = pd.DataFrame(columns=['x', 'y', 'z'])
        self.units = pd.DataFrame(columns=['x', 'y', 'z'])
        self.epoch_counter = pd.DataFrame(
            columns=['epoch_number', 'unit_number', 'w1', 'w2', 'w3', 'x1', 'x2', 'x3', 'net', 'w1_updated',
                     'w2_updated',
                     'w3_updated',
                     'length', 'w1_new', 'w2_new', 'w3_new', 'overall_weight_change'])
        self.learning_rate = learning_rate
        self.epochs_to_run = epochs_to_run
        self.classes = classes

        # checks
        self.check_fails = False

        # check if dataset file exists has correct extension
        if isinstance(dataset, str) and dataset.endswith('.xls') or dataset.endswith('.xlsx'):

            if (os.path.isfile(dataset)):
                self.dataset = pd.read_excel(dataset)
                # check if headers of data correct
                if set(['x', 'y', 'z']).issubset(self.dataset.columns):

                    self.dataset = self.normalise_data_frame(self.dataset)
                    self.write_to_file('NormalisedDataSet.xlsx', 'dataset')

                else:
                    self.check_fails = True
                    print('One or more columns x,y,z are missing')

            else:
                self.check_fails = True
                print('Dataset file does not exist.')
        else:
            self.check_fails = True
            print('Wrong file format for dataset.')

        # check if 2,3 or 4 value used for no of units unit
        if isinstance(no_of_units, int) and (no_of_units >= 2) and (no_of_units <= 4):

            if (self.classes is not None):
                if isinstance(self.classes, str):
                    # check if class file exists has correct extension
                    if self.classes.endswith('.xls') or self.classes.endswith('.xlsx'):

                        if (os.path.isfile(classes)):
                            classes = pd.read_excel(self.classes)
                            # check if headers of data correct
                            if set(['x', 'y', 'z']).issubset(classes.columns):
                                for i in range(0, no_of_units):
                                    x = classes.at[i, 'x']
                                    y = classes.at[i, 'y']
                                    z = classes.at[i, 'z']
                                    u = [x, y, z]
                                    self.units.loc[len(self.units)] = u
                            else:
                                self.check_fails = True
                                print('One or more columns x,y,z are missing')

                        else:
                            self.check_fails = True
                            print('Dataset file does not exist.')
                    else:
                        self.check_fails = True
                        print('Wrong file format for class set.')
                else:
                    self.check_fails = True
                    print('Wrong input for classes file.')

            else:
                # get range of x,y,z from range of table data
                self.setup_units(no_of_units)

        else:
            self.check_fails = True
            print('Number of units should be an integer from 2 to 4.')

        # exit program if parameters incorrect
        if (self.check_fails):
            exit()

        # if not (isinstance(dataset,pd.DataFrame):

        self.epoch_number = 0
        self.terminating_weight_change = terminating_weight_change

    # calculate length
    def euclidean_length(self, x, y, z):
        length = np.sqrt((x ** 2) + (y ** 2) + (z ** 2))
        return length

    def euclidean_distance(self, x1, y1, z1, x2, y2, z2):
        distance = np.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2) + ((z2 - z1) ** 2))
        return distance

    # normalise unit
    def normalise(self, x, y, z, length):
        x_prime = x / length
        y_prime = y / length
        z_prime = z / length
        return (x_prime, y_prime, z_prime)

    # check all points are length 1
    def do_check(self, x_prime, y_prime, z_prime):
        check = np.sqrt((x_prime ** 2) + (y_prime ** 2) + (z_prime ** 2))
        return check

        # normalize table

    def normalise_data(self, x, y, z):
        length = self.euclidean_length(x, y, z)
        normal = self.normalise(x, y, z, length)
        check = self.do_check(normal[0], normal[1], normal[2])
        return (length, normal[0], normal[1], normal[2], check)

    # normalise data frame
    def normalise_data_frame(self, df):
        # create columns for length, x_prime, y_prime, z_prime and check
        headers = ['x', 'y', 'z', 'length', 'x_prime', 'y_prime', 'z_prime', 'check']
        df = df.reindex(columns=headers)

        # calculate length, and normalize
        for index, row in df.iterrows():
            x = row[0]
            y = row[1]
            z = row[2]

            length, x_prime, y_prime, z_prime, check = self.normalise_data(x, y, z)
            df.at[index, 'length'] = length
            df.at[index, 'x_prime'] = x_prime
            df.at[index, 'y_prime'] = y_prime
            df.at[index, 'z_prime'] = z_prime
            df.at[index, 'check'] = check
        return df

    # get range of values for x,y,z
    def get_range(self):
        min_x = 0
        min_y = 0
        min_z = 0

        max_x = 0
        max_y = 0
        max_z = 0

        for index, row in self.dataset.iterrows():
            x = row[0]
            y = row[1]
            z = row[2]

            if x < min_x:
                min_x = x

            if y < min_y:
                min_y = y

            if z < min_z:
                min_z = z

            if x > max_x:
                max_x = x

            if y > max_y:
                max_y = y

            if z > max_z:
                max_z = z

        return min_x, max_x, min_y, max_y, min_z, max_z

    # get random x,y,z from the range of values in the dataset
    def random_vector(self, min_x, max_x, min_y, max_y, min_z, max_z):
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        z = np.random.uniform(min_z, max_z)
        return [x, y, z]

    # set up all units
    def setup_units(self, no_of_units):
        # get range of x,y,z from table data
        min_x, max_x, min_y, max_y, min_z, max_z = self.get_range()
        for i in range(0, no_of_units):
            u = self.random_vector(min_x, max_x, min_y, max_y, min_z, max_z)
            self.units.loc[len(self.units)] = u

    # calculate net and find closest unit
    def find_nearest(self, row):
        # calculate net
        nearest_index = 0
        nearest_net = (self.units['x_prime'][0] * row['x_prime']) + (self.units['y_prime'][0] * row['y_prime']) + (
                self.units['z_prime'][0] * row['z_prime'])
        net_list = []
        for index, row2 in self.units.iterrows():
            net = (row2['x_prime'] * row['x_prime']) + (row2['y_prime'] * row['y_prime']) + (
                    row2['z_prime'] * row['z_prime'])
            net_list.append(net)
            if (net > nearest_net):
                nearest_index = index
                nearest_net = net
            # print('net', net)
        # print(nearest_index)
        return nearest_index, net_list

    # update closest unit
    def update_nearest(self, nearest_index, row):
        nearest_value_x = self.units['x_prime'][nearest_index]
        updated_value_x = nearest_value_x + self.learning_rate * (row['x_prime'] - nearest_value_x)

        nearest_value_y = self.units['y_prime'][nearest_index]
        updated_value_y = nearest_value_y + self.learning_rate * (row['y_prime'] - nearest_value_y)

        nearest_value_z = self.units['z_prime'][nearest_index]
        updated_value_z = nearest_value_z + self.learning_rate * (row['z_prime'] - nearest_value_z)

        # print('updated_value', updated_value)
        return updated_value_x, updated_value_y, updated_value_z

    # normalize unit
    def normalise_nearest(self, updated_value_x, updated_value_y, updated_value_z):
        nearest_length, normalised_value_x, normalised_value_y, normalised_value_z, check = self.normalise_data(
            updated_value_x, updated_value_y, updated_value_z)
        return (nearest_length, normalised_value_x, normalised_value_y, normalised_value_z, check)

    # iterate through all data points

    def new_epoch(self):

        self.epoch_number += 1
        print(self.epoch_number)
        current_epoch = pd.DataFrame(
            columns=['epoch_number', 'unit_number', 'w1', 'w2', 'w3', 'x1', 'x2', 'x3', 'net', 'w1_updated',
                     'w2_updated',
                     'w3_updated',
                     'length', 'w1_new', 'w2_new', 'w3_new', 'overall_weight_change'])

        for index, row in self.dataset.iterrows():
            # print(row)
            self.units = self.normalise_data_frame(self.units)
            if (self.epoch_number == 1 and index == 0):
                self.write_to_file('NormalisedClasses.xlsx', 'classset')

            new_data_frame = pd.DataFrame(
                columns=['epoch_number', 'unit_number', 'w1', 'w2', 'w3', 'x1', 'x2', 'x3', 'net', 'w1_updated',
                         'w2_updated',
                         'w3_updated',
                         'length', 'w1_new', 'w2_new', 'w3_new', 'overall_weight_change'])

            for i in range(0, len(self.units)):
                # save epoch count
                new_data_frame.at[i, 'epoch_number'] = self.epoch_number
                # unit number
                new_data_frame.at[i, 'unit_number'] = i + 1

                # save w1,w2,w3(primes)
                new_data_frame.at[i, 'w1'] = self.units['x_prime'][i]
                new_data_frame.at[i, 'w2'] = self.units['y_prime'][i]
                new_data_frame.at[i, 'w3'] = self.units['z_prime'][i]

                # save data points
                new_data_frame.at[i, 'x1'] = self.dataset['x_prime'][index]
                new_data_frame.at[i, 'x2'] = self.dataset['y_prime'][index]
                new_data_frame.at[i, 'x3'] = self.dataset['z_prime'][index]

            # print(new_data_frame.to_string())

            # find nearest
            nearest_index, net_list = self.find_nearest(row)
            # update value with w+(n-x)
            updated_value_x, updated_value_y, updated_value_z = self.update_nearest(nearest_index, row)

            for i in range(0, len(self.units)):
                # save net
                new_data_frame.at[i, 'net'] = net_list[i]

                # copy over old values
                new_data_frame.at[i, 'w1_updated'] = new_data_frame.at[i, 'w1']
                new_data_frame.at[i, 'w2_updated'] = new_data_frame.at[i, 'w2']
                new_data_frame.at[i, 'w3_updated'] = new_data_frame.at[i, 'w3']

                # length of 1 for all
                new_data_frame.at[i, 'length'] = self.euclidean_length(new_data_frame.at[i, 'w1_updated'],
                                                                       new_data_frame.at[i, 'w2_updated'],
                                                                       new_data_frame.at[i, 'w3_updated'])

                # copy over old values
                normal = self.normalise(new_data_frame.at[i, 'w1_updated'], new_data_frame.at[i, 'w2_updated'],
                                        new_data_frame.at[i, 'w3_updated'], new_data_frame.at[i, 'length'])
                new_data_frame.at[i, 'w1_new'] = normal[0]
                new_data_frame.at[i, 'w2_new'] = normal[1]
                new_data_frame.at[i, 'w3_new'] = normal[2]

                # length of 1 for all

            # save updated values
            new_data_frame.at[nearest_index, 'w1_updated'] = updated_value_x
            new_data_frame.at[nearest_index, 'w2_updated'] = updated_value_y
            new_data_frame.at[nearest_index, 'w3_updated'] = updated_value_z

            # normalise value w/l
            length, normalised_value_x, normalised_value_y, normalised_value_z, check = self.normalise_nearest(
                updated_value_x,
                updated_value_y,
                updated_value_z)

            # save values to units
            self.units.at[nearest_index, 'x_prime'] = updated_value_x
            self.units.at[nearest_index, 'y_prime'] = updated_value_y
            self.units.at[nearest_index, 'z_prime'] = updated_value_z
            self.units.at[nearest_index, 'x'] = normalised_value_x
            self.units.at[nearest_index, 'y'] = normalised_value_y
            self.units.at[nearest_index, 'z'] = normalised_value_z

            # save length and normalised
            new_data_frame.at[nearest_index, 'length'] = length
            new_data_frame.at[nearest_index, 'w1_new'] = normalised_value_x
            new_data_frame.at[nearest_index, 'w2_new'] = normalised_value_y
            new_data_frame.at[nearest_index, 'w3_new'] = normalised_value_z

            current_data_frames = [current_epoch, new_data_frame]
            current_epoch = pd.concat(current_data_frames, ignore_index=True)

            # current_epoch.reset_index(drop=True)

            if (index == len(self.dataset) - 1):
                # print(current_epoch.at[0, 'w1'])

                for i in range(0, len(self.units)):
                    old_weight1 = (current_epoch.at[0 + i, 'w1'])
                    old_weight2 = (current_epoch.at[0 + i, 'w2'])
                    old_weight3 = (current_epoch.at[0 + i, 'w3'])
                    new_weight1 = (current_epoch.at[len(current_epoch) - len(self.units) + i, 'w1_new'])
                    new_weight2 = (current_epoch.at[len(current_epoch) - len(self.units) + i, 'w2_new'])
                    new_weight3 = (current_epoch.at[len(current_epoch) - len(self.units) + i, 'w3_new'])
                    euclidian_distance = self.euclidean_distance(old_weight1, old_weight2, old_weight3, new_weight1,
                                                                 new_weight2, new_weight3)
                    # print(self.epoch_number, euclidian_distance)
                    current_epoch.at[
                        len(current_epoch) - len(self.units) + i, 'overall_weight_change'] = euclidian_distance

                overall_epoch_frames = [self.epoch_counter, current_epoch]
                self.epoch_counter = pd.concat(overall_epoch_frames, ignore_index=True)

        # current_epoch = current_epoch.append(new_data_frame)
        # self.epoch_counter = self.epoch_counter.append(current_epoch)

        # print(self.epoch_counter.at[0, 'w1'])
        # print('here')
        # self.epoch_counter.at[len(self.epoch_counter - 1), 'overall_weight_change'] = 'beep'

    # check weight change
    def check_weight(self):
        terminate = True
        for i in range(0, len(self.units)):
            if ((self.epoch_counter.at[
                len(self.epoch_counter) - len(
                    self.units) + i, 'overall_weight_change']) >= self.terminating_weight_change):
                terminate = False
        return terminate

    # execute network
    def execute_network(self):

        # while weight change > terminating_weight_change, continue
        if (self.epochs_to_run == 0):
            terminate = False
            while (not terminate):
                self.new_epoch()
                terminate = self.check_weight()

        else:
            start = 0
            end = self.epochs_to_run

            while (start < end):
                self.new_epoch()
                start += 1

        # save output tables
        self.write_to_file('output.xlsx', 'epoch_counter')

        # plot graph and save it
        fig = plt.figure()
        # plots = []
        plot = fig.add_subplot(111)
        classes = []
        for i in range(0, len(self.units)):
            # plots.append(fig.add_subplot(111))
            x_points = []
            y_points = []
            for index, row in self.epoch_counter.iterrows():
                if (not np.isnan(row['overall_weight_change']) and row['unit_number'] == (i + 1)):
                    x_points.append(row['epoch_number'])
                    y_points.append(row['overall_weight_change'])

            classes.append(((i + 1), x_points, y_points))

        for i in classes:
            p = plot.plot(i[1], i[2], linestyle='--', marker='.', label=('Class' + str(i[0])))

        plot.set_xlabel('Epoch Number')
        plot.set_ylabel('Weight Change')
        plot.set_title('Kohonen Network with ' + str(len(self.units)) + ' units')

        plot.legend(bbox_to_anchor=(0.99, 1.1))

        graph_file_name = 'figure.png'
        plt.savefig(graph_file_name)

        plt.show()
        fig.show()

    def write_to_file(self, output_file_name, dataframe):
        writer = pd.ExcelWriter(output_file_name)

        if (dataframe == 'epoch_counter'):
            self.epoch_counter.to_excel(writer, 'Sheet1', index=False)

        elif (dataframe == 'dataset'):

            self.dataset.to_excel(writer, 'Sheet1', index=False)

        elif (dataframe == 'classset'):
            self.units.to_excel(writer, 'Sheet1', index=False)

        writer.save()
