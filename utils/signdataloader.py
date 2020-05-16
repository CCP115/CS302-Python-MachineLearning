from __future__ import print_function, division
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset

# On Cecil's Local Machine
# csv_file="/data/sign_language_mnist/sign_mnist_train.csv"
# root_dir="/data/sign_language_mnist"

class SignDataLoader():
    def __init__(self, csv_file, root_dir):
        self.sign_frame = pd.read_csv(csv_file)
        self.train_shffl = random.sample(list(range(27455)), 27455)
        self.test_shffl = random.sample(list(range(7172)), 7172)
        self.train_shffl_idx = 0
        self.test_shffl_idx = 0


    def __getitem__(self, idx):
        """
        Returns the indexed item in dict
        letter : str, image : 28x28 float array
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        num = self.sign_frame.iloc[idx, 0]
        num = torch.from_numpy(np.array(num, dtype='float'))

        image = self.sign_frame.iloc[idx, 1:]
        image = np.array([image])
        image = torch.from_numpy(image.astype('float').reshape(-1, 28))
        sample = {'data': image, 'target': num}

        return sample


    def get_batch(self, size, idx):
        """
        Returns a batch of given size in two lists
        Inputs
        size: length of two lists returned
        idx: item to retrieve beginning at
        Outputs
        targets: list of length size, contains ground truths for the images
        imgdata: list of length size, contains greyscale image data
        """
        targets = []
        imgdata = []

        for i in range(size):
            sample = self.__getitem__(idx)
            targets.append(sample['target'])
            imgdata.append(sample['data'])

        return targets, imgdata


    def get_shuffled_batch(self, size, test=0):
        """
        Returns a batch of given size in two lists, as well as exhausted flag
        Inputs
        size: length of two lists returned
        test: determines whether to get train data or test data, 0 for train, 1 for test
        Outputs
        targets: list of length size, contains ground truths for the images
        imgdata: list of length size, contains greyscale image data
        exhausted: returns true if all data has been iterated through
        """
        targets = []
        imgdata = []

        if test == 0: # Training data
            # Get an image and label to fit size
            for i in range(size):
                # Use train_shffl_indx to get shuffled indx from train_shffl, and then __getitem__
                sample = self.__getitem__(self.train_shffl[self.train_shffl_idx])
                # Append target and imgdata
                targets.append(sample['target'])
                imgdata.append(sample['data'])
                # Break if train_shffl_idx is at max index, 27455, and reset train_shffl_idx
                if self.train_shffl_idx == 27454:
                    self.train_shffl_idx = 0

                    # Stack imgdata and targets into tensors from a list
                    imgdata = torch.stack(imgdata[:], dim=0)
                    targets = torch.stack(targets[:], dim=0)

                    return targets, imgdata, True
                else: # Increment the train_shffl_idx
                    self.train_shffl_idx = self.train_shffl_idx + 1

        elif test == 1: # Testing data
            for i in range(size):
                # Use test_shffl_indx to get shuffled indx from test_shffl, and then __getitem__
                sample = self.__getitem__(self.test_shffl[self.test_shffl_idx])
                # Append target and imgdata
                targets.append(sample['target'])
                imgdata.append(sample['data'])
                # Break if test_shffl_idx is at max index, 7171, and reset test_shffl_idx
                if self.test_shffl_idx == 7171:
                    self.test_shffl_idx = 0

                    # Stack imgdata and targets into tensors from a list
                    imgdata = torch.stack(imgdata[:], dim=0)
                    targets = torch.stack(targets[:], dim=0)

                    return targets, imgdata, True
                else:  # Increment the test_shffl_idx
                    self.test_shffl_idx = self.test_shffl_idx  +1

        else:
            print("Error, incorrect mode passed.")

        # Stack imgdata and targets into tensors from a list
        imgdata = torch.stack(imgdata[:], dim=0)
        targets = torch.stack(targets[:], dim=0)

        return targets, imgdata, False


    def get_shffl_indx(self):
        """
        Returns the shuffle indexes used to iterate through the shuffled arrays
        indexes: Dict containing train and test shuffle indexes, respectively
        """
        indexes = {"Train_Shffl_Idx": self.train_shffl_idx, "Test_Shffl_Idx": self.test_shffl_idx}
        return indexes


    def get_shffl(self, mode=0):
        """
        Returns the shuffled array itself, with shuffled indexes
        mode: 0 means return training index array, 1 means returns testing index array
        shuffled: Shuffled array of indexes for pulling imagedata from the dataframe
        """
        if mode == 0:
            return self.train_shffl
        elif mode == 1:
            return self.test_shffl
        else:
            return 0


    def show_item(self, idx):
        """
        Creats a matplotlib graph showing the sign language image
        28x28 resolution
        letter on the x-axis
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = ASL[self.sign_frame.iloc[idx, 0]]
        image = self.sign_frame.iloc[idx, 1:]
        image = np.array([image])
        image = image.astype('float').reshape(-1, 28)
        plt.xlabel("The Letter shown above is: " + img_name)
        plt.imshow(image)
        plt.pause(0.001)
        plt.show()
        return


    def set_train_shffl_idx(self, idx):
        """
        Sets the train_shffl_idx to specified value
        """
        self.train_shffl_idx = idx


    def set_test_shffl_idx(self, idx):
        """
        Sets the test_shffl_idx to specified value
        """
        self.test_shffl_idx = idx


    def get_len(self):
        return self.sign_frame.shape[0]