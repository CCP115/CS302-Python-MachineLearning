"""
Main.py
Executes, trains, tests, and displays results for all four models
"""

# Python and PyTorch imports
from __future__ import print_function
import torch
import torch.optim as optim # contains different optimizers
from torch.optim.lr_scheduler import StepLR

import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd

# Custom Imports
from models.model1 import model1
from models.model2 import model2
from models.model3 import model3
from models.model4 import model4
from utils.signdataloader import SignDataLoader
from utils.user_input import user_input, print_user_input
from utils.evaluation import plot_acc_loss, plot_confusion_matrix, plot_classification_report

# The ASL Tuple is for translating a number to a letter, e.g. 1 is a, 2 is b, etc.
ASL = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
       'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
       's', 't', 'u', 'v', 'w', 'x', 'y', 'z')


"""
Main Function
"""
def main():
    # Get user input on hyperparameters
    hyper_params = user_input()

    # Edit parameters here
    model_num = hyper_params["model_number"]
    epochs = hyper_params["epochs"]
    train_batch_size = hyper_params["train_batch_size"]
    test_batch_size = hyper_params["test_batch_size"]
    learning_rate = hyper_params["learning_rate"]
    gamma = hyper_params["gamma"]
    log_interval = hyper_params["log_interval"]

    # Print hyperparameters for user to see
    print_user_input(hyper_params)

    # Set manual seed
    torch.manual_seed(1)

    # Check whether you can use Cuda
    use_cuda = torch.cuda.is_available()
    # Use Cuda if you can
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device: ", device)

    # Train based on hyperparameters
    train_cnn(model_num, epochs, train_batch_size, test_batch_size, learning_rate, gamma, log_interval, device)


"""
Code copied and modified from MNIST Lab
Training Function
"""
def train_cnn(model_num, epochs, train_batch_size, test_batch_size, learning_rate, gamma, log_interval, device):
    # Get dataset using custom class
    train_dataset = SignDataLoader(csv_file="data/sign_language_mnist/sign_mnist_train.csv",
                                   root_dir="data/sign_language_mnist")
    test_dataset = SignDataLoader(csv_file="data/sign_language_mnist/sign_mnist_test.csv",
                                  root_dir="data/sign_language_mnist")

    # Create network based on model_num
    model = create_network(model_num, device)

    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    # Create lists to store loss for each train and test epoch
    train_loss = []
    test_loss = []
    # Create lists to store accuracy for each epoch test
    train_acc = []
    test_acc = []

    # Train and test for specified epochs
    for epoch in range(1, epochs + 1):
        # Train model for each epoch
        train(model, optimizer, epoch, train_batch_size, device, train_dataset, log_interval, train_loss, train_acc)
        # Test model at end of every epoch
        test(model, test_batch_size, device, test_dataset, test_loss, test_acc)
        scheduler.step()

    # Do final test and graph confusion matrix
    final_test(model, test_batch_size, device, test_dataset, test_acc)

    # Plot accuracy vs loss graph
    plot_acc_loss(epochs, train_loss, test_loss, train_acc, test_acc)


def train(model, optimizer, epoch, batch_size, device, dataset, log_interval, train_loss, train_acc):
    """
    Code copied and modified from MNIST Lab
    Training Function
    """
    # Iterate through the dataset until it is exhausted
    exhausted = False
    batch_idx = 0
    sum_loss = 0
    correct = 0

    while exhausted == False:
        # Pull a batch
        label, imgdata, exhausted = dataset.get_shuffled_batch(batch_size, test=0)

        # Send image data to device
        imgdata, label = imgdata.to(device, dtype=torch.float), label.to(device, dtype=torch.float)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(imgdata)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(label.view_as(pred)).sum().item()
        loss = F.cross_entropy(output, label.long())
        loss.backward()
        optimizer.step()

        # Sum losses for all batches
        sum_loss = sum_loss + loss.item()

        # Display information
        batch_idx = batch_idx + 1
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(imgdata),
                dataset.get_len(),
                100 * (batch_idx * len(imgdata) / dataset.get_len()),
                loss.item()))

    # Calculate mean loss for current epoch
    train_loss.append(sum_loss / batch_idx)
    # Calculate accuracy of current epoch
    train_acc.append(100. * correct / dataset.get_len())


def test(model, batch_size, device, dataset, loss_arr, acc):
    """
    Code copied and modified from MNIST Lab
    Testing Function
    """
    model.eval()
    test_loss = 0
    correct = 0
    exhausted = False

    with torch.no_grad():
        while exhausted == False:
            # Pull a batch
            label, imgdata, exhausted = dataset.get_shuffled_batch(batch_size, test=1)

            # Send image data to device
            imgdata, label = imgdata.to(device, dtype=torch.float), label.to(device, dtype=torch.float)

            output = model(imgdata)
            test_loss += F.cross_entropy(output, label.long(), reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= dataset.get_len()

    # Save loss value for batch
    loss_arr.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,
        correct,
        dataset.get_len(),
        100. * correct / dataset.get_len()))

    # Save accuracy of each test epoch
    acc.append(100. * correct / dataset.get_len())


def final_test(model, batch_size, device, dataset, acc):
    """
    Code copied and modified from MNIST Lab
    Final Testing Function
    Copied from test(), but with added graphing functionality
    """
    model.eval()
    test_loss = 0
    correct = 0
    exhausted = False

    model_preds = []
    model_targs = []

    with torch.no_grad():
        while exhausted == False:
            # Pull a batch
            label, imgdata, exhausted = dataset.get_shuffled_batch(batch_size, test=1)

            # Send image data to device
            imgdata, label = imgdata.to(device, dtype=torch.float), label.to(device, dtype=torch.float)

            # Evaluate model
            output = model(imgdata)

            # Calculate loss and accuracy
            test_loss += F.cross_entropy(output, label.long(), reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(label.view_as(pred)).sum().item()

            # Store predictions and targets
            model_preds.append(pred)
            model_targs.append(label.view_as(pred))

    test_loss /= dataset.get_len()

    print('\nFinal Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,
        correct,
        dataset.get_len(),
        100. * correct / dataset.get_len()))

    # Modify targets and predictions into cpu based numpy arrays
    model_preds = torch.cat(model_preds)
    model_targs = torch.cat(model_targs)
    model_preds = model_preds.cpu()
    model_targs = model_targs.cpu()
    model_preds = np.array(model_preds)
    model_targs = np.array(model_targs)

    # Plot confusion matrix
    plot_confusion_matrix(model_targs, model_preds, ASL)

    # Plot table of classification scores
    plot_classification_report(model_targs, model_preds, ASL)


def create_network(model_num, device):
    """
    create_network creates a model of the given number and sends it to the given device
    Returns:
        model: Model created with given parameters
    """
    if model_num == 1:
        model = model1().to(device)
    elif model_num == 2:
        model = model2().to(device)
    elif model_num == 3:
        model = model3().to(device)
    elif model_num == 4:
        model = model4().to(device)


    return model


if __name__ == '__main__':
    main()
