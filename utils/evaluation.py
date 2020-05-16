"""
Contains Python code for different graphing functions used throughout our code
Written by Cecil Symes
csym531
"""

# Imports
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix(model_targs, model_preds, ASL):
    """
    plot_confusion_matrix plots a confusion matrix
    Inputs:
        model_targs: Ground truths
        model_preds: What the model predicted
        ASL: A Tuple containing the corresponding letter for each number
    """
    # Create confusion matrix
    cm = confusion_matrix(model_preds, model_targs)

    # Converting to panda dataframe
    df = pd.DataFrame(cm, range(24), range(24))

    df.rename(columns=lambda a: ASL[a],
              index=lambda a: ASL[a],
              inplace=True)

    sn.set(font_scale=1)  # for label size
    ax = sn.heatmap(df, annot=True, annot_kws={"size": 10}, fmt="d", linewidths=.5, xticklabels=1, yticklabels=1)
    ax.set(xlabel="Predicted Value", ylabel="Ground Truth")
    #plt.show()


def plot_acc_loss(epochs, train_loss, test_loss, train_acc, test_acc):
    """
    plot_acc_loss plots a figure with two subplots.
    One subplot is loss against number of epochs, with a line for test and train datasets.
    Other subplot is accuracy against number of epochs, with a line for test and train datasets.
    Args:
        epochs: Number of epochs trained and tested for
        train_loss: 1D Array of average loss per training epoch, with index corresponding to epoch
        test_loss: 1D Array of average loss per testing epoch, with index corresponding to epoch
        train_acc: 1D Array of average accuracy per training epoch, with index corresponding to epoch
        test_acc: 1D Array of average accuracy per testing epoch, with index corresponding to epoch
    """

    # Set x axis markings
    x = np.arange(0, epochs, 1)
    # Create figure and axis and subplots
    fig, (ax1, ax2) = plt.subplots(2)
    # Plot data
    ax1.plot(x, train_loss, marker='x', label='Training Loss')
    ax1.plot(x, test_loss, marker='.', label='Testing Loss')
    ax2.plot(x, train_acc, marker='x', label='Train Accuracy')
    ax2.plot(x, test_acc, marker='.', label='Test Accuracy')
    # Change axis labels
    ax1.set(ylabel='Average Loss')
    ax2.set(xlabel='Epochs', ylabel='% Accuracy')
    # Change axes limits
    ax1.axis([0, epochs, 0, max(max(train_loss), max(test_loss))])
    ax2.axis([0, epochs, 0, 100])
    # Change spacing between axis markings
    ax1.set_xticks(np.arange(len(train_loss)))
    ax1.set_xticklabels(np.arange(1, len(train_loss) + 1))
    ax2.set_xticks(np.arange(len(train_loss)))
    ax2.set_xticklabels(np.arange(1, len(train_loss) + 1))
    # Title
    ax1.set_title('Average Loss across Epochs')
    ax2.set_title('% Accuracy across Epochs')
    # Annotating the coordinates of each point on the graph
    idx = 0
    while idx < len(train_loss):
        ax1.annotate("{:.2f}".format(train_loss[idx]), (x[idx],train_loss[idx]), textcoords="offset points",
                     xytext=(0,10), ha='center')
        ax1.annotate("{:.2f}".format(test_loss[idx]), (x[idx],test_loss[idx]), textcoords="offset points",
                     xytext=(0,10), ha='center')
        ax2.annotate("{:.2f}".format(train_acc[idx]), (x[idx],train_acc[idx]), textcoords="offset points",
                     xytext=(0,10), ha='center')
        ax2.annotate("{:.2f}".format(test_acc[idx]), (x[idx],test_acc[idx]), textcoords="offset points",
                     xytext=(0,10), ha='center')
        idx = idx + 1

    # Legend
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    plt.show()


def plot_classification_report(true, preds, ASL):
    """
    plot_classification_report calls the sk.learnmetrics function classification_report and plots the results
    Inputs:
        model_targs: Ground truths
        model_preds: What the model predicted
    """
    # Create dict of classification report
    result_dict = classification_report(true, preds, output_dict=True)
    # Set row and column labels
    row_label = list(ASL)
    row_label.remove('z')
    row_label.remove('j')
    col_label = ['Precision', 'Recall', 'F1-Score', 'Support']
    # data is a list of lists, each index contains the scoring for the corresponding index in headers
    data = []
    temp = []

    # Extract dict values
    for outer_key in result_dict:
        if outer_key != 'accuracy':
            for inner_key in result_dict[outer_key]:
                temp.append(str(result_dict[outer_key][inner_key]))
            data.append(temp)
            temp = []
        else:
            break

    fig = plt.figure()
    ax = fig.add_subplot(111)

    print("col label", col_label)
    print("row label", row_label)
    print("data", data)


    the_table = plt.table(cellText=data,
                      colWidths=[0.1] * (len(col_label) + 1),
                      rowLabels=row_label,
                      colLabels=col_label,
                      loc='center right')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    the_table.scale(2, 1)
    ax.axis('off')
    ax.axis('tight')
    #plt.show()


def show_batch(sample_batched):
    """Show images for a batch of samples."""
    images_batch, targets_batch = \
        sample_batched['data'], sample_batched['target']
    batch_size = len(images_batch)

    for i in range(batch_size):
        ax = plt.subplot(2, 5, i + 1)
        plt.tight_layout()
        ax.set_title('Target - {}'.format(targets_batch[i]))
        ax.axis('off')
        plt.imshow(images_batch[i])
