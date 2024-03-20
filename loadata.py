import pandas as pd
import numpy as np
import os
import progressbar
from config import Config
from random import shuffle
import math
from utils.padding import padding
from utils.augment import augment
import openpyxl

class Person:
    def __init__(self, filepath, cfg: Config, train=True, condition="patient", rp=False):
        """
        :param filepath:
                fp , e.g., "dataset/data/xxxxx.xlsx"
        :param cfg:
                config file, useful in reading labels and split gait cycles ....
        """
        self.train = train
        self.use_CoM = cfg.use_CoM
        self.padding = cfg.padding
        self.augment = cfg.augment
        self.mass_center_features = None
        self.max_frames = 0
        self.min_frames = 1e4
        self.frames = 0
        self.labels = list()
        self.features = list()
        self.sheet = list()
        self.name = None
        self.fp = filepath
        self.cfg = cfg
        self.condition = condition
        self.rp = rp                                                                    # whether use relative displacement or not
        self.extract(cfg.time_split)

    def aug(self, features, labels):
        out = augment(features)
        for aug_data in out:
            self.features.append(aug_data)
            self.labels.append(labels)

        self.frames += len(out) * features.shape[1]

    def extract(self, time_split):
        filename = self.fp.split('/')[-1].split('.')[0]
        self.name = filename.split(' ')[0]                                              # in this way, we can directly match the label
        cyc_sheet = pd.read_excel(self.fp, sheet_name="Markers", engine='openpyxl')     # record important gait cycle time points

        drop_labels = ["Frame"]
        if self.cfg.ignore_spine:
            for seg in self.cfg.spine_segment:
                for coord in ["x", "y", "z"]:
                    drop_labels.append(seg + " " + coord)

        for sheet_id in self.cfg.segment_sheet_idx:
            sheet_name = self.cfg.segment_sheets[sheet_id]
            sheet = pd.read_excel(self.fp, sheet_name=sheet_name)
            if sheet_id != 6:                                                           # Joint Angles sheet is not same as segments sheet
                sheet = sheet.drop(drop_labels, axis=1)
            else:
                sheet = sheet.drop(["Frame"], axis=1)
            self.sheet.append(sheet)

        rowId = int(self.name[1:])                                                      # filter 'S'
        labels = self.cfg.labels.loc[rowId - 1][2:]                                     # filter timestamp and name
        universe_labels = [int(labels[t][-1]) - 1 for t in range(len(labels))]          # multiple gaitcycles share the same label

        # if self.cfg.use_CoM:
        #     try:
        #         # some of the exported files don't have this working sheet at first
        #         mass_center_sheet = pd.read_excel(self.fp, sheet_name="Center of Mass")
        #         mass_center_sheet = mass_center_sheet.drop(["Frame"], axis=1)
        #         self.mass_center_features = mass_center_sheet.to_numpy()  # [frames, joints * 3]
        #         self.use_CoM = True
        #     except:
        #         # accidentally find that there are some files don't contain this worksheet
        #         self.use_CoM = False
        #         print("{} doesn't have sheet Center of Mass\n".format(self.fp))

        for start in range(0, len(cyc_sheet), time_split):
            if start + 4 < len(cyc_sheet):
                end = start + 4
                interval_start = cyc_sheet['Frame'][start]
                interval_end = cyc_sheet['Frame'][end]
                self.max_frames = max(interval_end - interval_start, self.max_frames)
                self.min_frames = min(self.min_frames, interval_end - interval_start)
                self.frames += interval_end - interval_start

                features = None

                for sheetid, sheet in enumerate(self.sheet):
                    x_sheet_features = sheet.to_numpy()                                 # in shape [frames, joints * 3(3D space)]

                    """
                    for position information, use relative displacement instead
                    """
                    if sheetid == 0 and self.rp:
                        block = np.zeros_like(x_sheet_features)
                        block[1:, :] = x_sheet_features[:-1, :]
                        x_sheet_features = x_sheet_features - block

                    x_sheet_features = x_sheet_features[interval_start:interval_end]
                    x_sheet_features = np.reshape(x_sheet_features, (x_sheet_features.shape[0], -1, 3))
                    
                    """
                    some people are walking circularly, so use absolute values to rectify the direction on x dimension.
                    """
                    if sheetid == 0 or sheetid == 1:
                        x_sheet_features[:, :, 0] = np.abs(x_sheet_features[:, :, 0])
                    x_sheet_features = np.transpose(x_sheet_features, (2, 0, 1))        # [channel, frames, joints]
                    features = x_sheet_features if features is None else \
                        np.concatenate([features, x_sheet_features], axis=0)            # along channels dimension

                if self.use_CoM:
                    com_features = self.mass_center_features[interval_start:interval_end]
                    com_features = np.reshape(com_features, (com_features.shape[0], 1, -1))
                    com_features = np.transpose(com_features, (2, 0, 1))
                    channel, frame, joint = com_features.shape
                    channel_1 = features.shape[0]
                    if channel != channel_1:
                        if channel >= channel_1:
                            com_features = com_features[:channel_1]
                        else:
                            # use zero-padding to enlarge the channels
                            zero_blocks = np.zeros((channel_1 - channel, frame, joint))
                            com_features = np.concatenate([com_features, zero_blocks], axis=0)

                    # take "Center of Mass" as a new joint
                    features = np.concatenate([features, com_features], axis=-1)

                # use the [affected, unaffected] layout to replace original [left_joints, right_joints]
                if self.condition == "patient" and self.cfg.affected_side[rowId].split('-')[0] == 'R':
                    for i in range(0, len(self.cfg.sides), 2):
                        features[:, :, self.cfg.sides[i]] = features[:, :, self.cfg.sides[i + 1]]

                # padding the data
                features = padding(data=features, avg=self.cfg.avg_frames) if self.padding else features

                self.features.append(features)
                self.labels.append(universe_labels)

        cycles = len(self.features)

        # test data does not need data augmentation
        if self.train:
            # augmentation 1: moving average
            if self.padding:                                                            # if not padding, each gait cycle of the same person is not 100% same
                for i in range(cycles):
                    for j in range(i + 4, cycles, 4):
                        avg_gait_features = (self.features[i] + self.features[j]) / 2
                        self.features.append(avg_gait_features)
                        self.labels.append(universe_labels)
                        self.frames += avg_gait_features.shape[1]                       # [channel, frame, joint]

            cycles = len(self.features)                                                 # update cycles to avoid infinite recursive

            # augmentation 2: jittering, scaling
            if self.augment:
                for i in range(cycles):
                    self.aug(self.features[i], universe_labels)


class Dataset:
    def __init__(self, cfg: Config, dataset_path="./dataset/data", padding=True, seed=0, rp=True):
        """
        :param cfg:
            a Config object
        :param dataset_path:
            dataset path
        :param padding:
            whether pad the gait cycles to perform batch training

        what we are going to do here is to find the maximum time frames of the dataset,
        then do interpolation to achieve batch training.  (firstly, simply use linear xx)

        another thing is to split the dataset into a train & test set.
        """
        self.train_ptr = 0
        self.test_ptr = 0
        self.batch_index = 0
        # since the frames spent on each gait cycle are different between different people,
        # the following frames variable are recorded for statistics and padding
        self.maxFrame = 0
        self.minFrame = 1e4
        self.total_frames = 0
        self.train_data = list()
        self.train_label = list()
        self.test_data = list()
        self.test_label = list()

        # for debugging overfitting problem
        self.train_name = list()
        self.test_name = list()
        self.train_cycle_index = list()
        self.test_cycle_index = list()

        self.dspath = dataset_path
        self.cfg = cfg

        self.rp = rp

        if self.cfg.use_CoM:
            self.cfg.nodes.append("Center of Mass")

        np.random.seed(seed)

        # since the dataset is imbalanced, if we don't load it in this way.
        # the model may only be trained on the patient data
        # categories = ["healthy", "patient"]
        categories = ["patient", "healthy"]

        for category in categories:
            path = os.path.join(self.dspath, category)
            total_files = os.listdir(path)
            total_len = len(total_files)
            train_num = math.floor(total_len * self.cfg.train_rate)
            train_set = np.random.choice(total_len, train_num, replace=False)
            print("Start loading {} files:".format(category))
            p = progressbar.ProgressBar()
            for i in p(range(total_len)):
                fp = os.path.join(path, total_files[i])
                if i not in train_set:
                    self.load_person(fp, train=False, condition=category)
                else:
                    self.load_person(fp, train=True, condition=category)

        actual_train_num = len(self.train_data)
        actual_test_num = 0
        for test_data in self.test_data:
            actual_test_num += len(test_data)
        actual_num = actual_test_num + actual_train_num

        self.train_batches = actual_train_num / self.cfg.batch_size
        self.test_batches = actual_test_num / self.cfg.batch_size

        print("Finish loading.")
        print("Train Set: {}, Test Set: {}".format(len(self.train_data), len(self.test_data)))
        print("Minimum Frame of one gait cycle is {}".format(self.minFrame))
        print("Maximum Frame of one gait cycle is {}".format(self.maxFrame))
        print("Average Frame of one gait cycle is {}".format(self.total_frames / actual_num))

        # self.zero_padding()
        # add the "batch" dimension to fit the model requirement
        self.extend()

    def extend(self):
        for i in range(len(self.train_data)):
            self.train_data[i] = self.train_data[i][None, :, :, :]

        # refer to majority vote , so it would be clear that test_data is actually a list of lists
        for i in range(len(self.test_data)):
            for j in range(len(self.test_data[i])):
                self.test_data[i][j] = self.test_data[i][j][None, :, :, :]

    def load_person(self, fp, train=True, condition="patient"):
        p = Person(fp, self.cfg, train=train, condition=condition, rp=self.rp)
        self.maxFrame = max(self.maxFrame, p.max_frames)  # for interpolation
        self.minFrame = min(self.minFrame, p.min_frames)
        self.total_frames += p.frames

        cycles = len(p.features)

        if train:
            for features, labels in zip(p.features, p.labels):
                self.train_data.append(features)
                self.train_label.append(labels)
            for i in range(cycles):
                self.train_name.append(p.name)
                self.train_cycle_index.append(i + 1)
        else:
            self.test_data.append(p.features)
            # since all the gait cycles are coming from the same person, share the same label
            self.test_label.append(p.labels[0])
            self.test_name.append(p.name)

            # cycle index, in this case, I predict the label of each gait cycle of the same person
            # However, now I want to use vote method to determine the final result
            # self.test_cycle_index.append(i + 1)

    def load_batch_data_train(self, items: list, batchsize=4):
        bat_label = list()
        bat_data = list()
        train_len = len(self.train_data)

        # single item test first, to see whether this padding and interpolation method is valid
        next_bc = min((self.batch_index + 1) * batchsize, train_len)
        for item in items:
            item_label = list()
            item_index = item - 1
            for labels in self.train_label[self.batch_index * batchsize:next_bc]:
                item_label.append(labels[item_index])
            bat_label.append(item_label)

        bat_data = np.asarray(self.train_data[self.batch_index * batchsize:next_bc], dtype=np.float32)
        bat_data = bat_data.squeeze(axis=1)
        # bat_label = np.asarray(bat_label)

        self.batch_index = self.batch_index + 1 if next_bc < train_len else 0

        return bat_data, bat_label

    def load_data(self, items: list, train=True):
        """
        one gait cycle once time call this function.

        Notably, when loading test data, it returns a list of gait cycles from the same person.
        then, we use majority vote method to determine the final result
        """
        data = None
        label = list()

        # for tracking overfitting problem
        patient_name = None
        cycle_index = None

        if train:
            patient_name = self.train_name[self.train_ptr]
            cycle_index = self.train_cycle_index[self.train_ptr]
            data = self.train_data[self.train_ptr]
            for item in items:
                # start from 0
                item_label = list()
                item_index = item - 1
                # in this way, we can ensure each label has an extra "batch = 1" dimension, which is more flexible
                item_label.append(self.train_label[self.train_ptr][item_index])
                label.append(item_label)
            self.train_ptr += 1
            self.train_ptr = 0 if self.train_ptr == len(self.train_data) else self.train_ptr

        else:
            patient_name = self.test_name[self.test_ptr]
            data = self.test_data[self.test_ptr]  # Still a list, consisted of several gait cycles from same person
            for item in items:
                # start from 0
                item_index = item - 1
                label.append(self.test_label[self.test_ptr][item_index])
            self.test_ptr += 1
            self.test_ptr = 0 if self.test_ptr == len(self.test_data) else self.test_ptr

        return data, label, patient_name, cycle_index

    def shuffle(self):
        """
        :return:
                don't try to directly shuffle on "self.train_data" since we may not get the correct label.
                Thus, shuffle the index and use a new container to replace the original one

                don't try to directly operate on source container to avoid data overwriting or data missing
        """
        indexs = list([i for i in range(len(self.train_data))])
        shuffle(indexs)
        new_train_data = list()
        new_label_data = list()

        new_train_name = list()
        new_train_cindex = list()

        for j in indexs:
            new_train_data.append(self.train_data[j])
            new_label_data.append(self.train_label[j])

            new_train_name.append(self.train_name[j])
            new_train_cindex.append(self.train_cycle_index[j])

        self.train_label = new_label_data
        self.train_data = new_train_data

        self.train_name = new_train_name
        self.train_cycle_index = new_train_cindex

        self.train_ptr = 0

# code for debugging
# a = Person("dataset/data/S6 C-002.xlsx")
# D = Dataset()
