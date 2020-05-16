"""
user_input Python Package
Written by Cecil Symes
Contains the user_input function
"""


def get_model_number():
    """
    get_model_number asks user for which model to train
    Returns:
        modelNum: integer of which model will be trained
    """
    modelNum = ""

    while (modelNum == "") or (int(modelNum) < 1) or (int(modelNum) > 4):
        print('Which model would you like to test and train?')
        if (modelNum == "") or (int(modelNum) < 1) or (int(modelNum) > 4):
            print('Please enter a number between 1 & 4 inclusive.')
        modelNum = input('> ')
    print("You are training model {}.".format(modelNum))
    return int(modelNum)


def get_epochs(def_epochs):
    """
    get_epochs asks user for number of epochs to train for
    Returns:
        epochs: integer of number of epochs to train for
    """

    epochs = ""
    while (epochs == "") or (int(epochs) < 1):
        print('How many epochs would you like to train for?')
        print('Press Enter for default number of {} epoches.'.format(def_epochs))
        epochs = input('> ')
        if epochs == "":
            epochs = def_epochs
            break
        elif int(epochs) > 1:
            break
        elif int(epochs) < 1:
            print('Please enter a valid number of epochs over 0.')
    print('Training for {} epochs.'.format(epochs))
    print('A test will be conducted after each epoch for Loss and Accuracy milestones.')
    return int(epochs)


def get_train_batch_size(def_batch_size):
    """
    get_batch_size asks user for batch size to user for training
    Returns:
        batch_size: integer of size to user for batches during network training
    """
    batch_size = ""
    while (batch_size == "") or (int(batch_size) < 1):
        print('What batch size would you like to train with?')
        print('Press Enter for default training batch size of {} images.'.format(def_batch_size))
        batch_size = input('> ')
        if batch_size == "":
            batch_size = def_batch_size
            break
        elif int(batch_size) > 0:
            break
        elif int(batch_size) < 1:
            print('Please enter a valid number over 0.')
    print('Training with a batch size of {} images.'.format(batch_size))
    return int(batch_size)


def get_test_batch_size(def_batch_size):
    """
    get_batch_size asks user for batch size to user for testing
    Returns:
        batch_size: integer of size to user for batches during network testing
    """
    batch_size = ""
    while (batch_size == "") or (int(batch_size) < 1):
        print('What batch size would you like to test with?')
        print('Press Enter for default test batch size of {} images.'.format(def_batch_size))
        batch_size = input('> ')
        if batch_size == "":
            batch_size = def_batch_size
            break
        elif int(batch_size) > 0:
            break
        elif int(batch_size) < 1:
            print('Please enter a valid number over 0.')
    print('Testing with a batch size of {} images.'.format(batch_size))
    return int(batch_size)


def get_learning_rate(def_lr):
    """
    get_learning_rate gets user input of what learning rate the model should use
    Returns:
        lr: float representing learning_rate to use
    """
    lr = ""
    while (lr == "") or (float(lr) < 0) or (float(lr) > 1):
        print('What learning rate would you like to train with?')
        print('Press Enter for default learning rate of {}.'.format(def_lr))
        lr = input('> ')
        if lr == "":
            lr = def_lr
            break
        elif (float(lr) > 0) and (float(lr) < 1):
            break
        elif float(lr) < 0:
            print('Please enter a valid learning rate between 0 and 1.')
        elif float(lr) > 1:
            print('Learning rate is too large, please enter a valid learning rate between 0 and 1.')
    print('Training with a learning rate of {}.'.format(lr))
    return float(lr)


def user_input():
    """
    user_input function asks user for input on which model to train, num of batches, num of epochs, and learning rate
    Returns:
        dict containing:
        model_num: Number of model to train and test
        epochs: Number of epochs to train for
        train_batch_size: Number of images per batch used for training
        test_batch_size: Number of images per batch used for testing
        learning_rate: Learning rate of model
        gamma: Gamma (?)
        log_interval: log_interval
    """

    # Set Master Defaults
    def_epochs = 4 # 4 allows for quick training
    def_batch_size = 10 # Default batch size testing and training
    def_lr = 0.001 # Default learning rate from MNIST lab and also Model 2 article
    # Default from MNIST lab, haven't found sufficient reason to change
    def_gamma = 0.7
    def_log_interval = 10

    # Get model number from user
    modelNum = get_model_number()

    # Model specific defaults
    if modelNum == 1:
        def_epochs = 10
    elif modelNum == 2:
        def_epochs = 50
        def_batch_size = 100
    elif modelNum == 3:
        def_epochs = 10
        def_batch_size = 10
    elif modelNum == 4:
        def_epochs = 10
        def_batch_size = 10

    # Get number of epochs from user
    epochs = get_epochs(def_epochs)
    # Get training batch size from user
    train_batch_size = get_train_batch_size(def_batch_size)
    # Get testing batch size from user
    test_batch_size = get_test_batch_size(def_batch_size)
    # Get learning rate from user
    lr = get_learning_rate(def_lr)

    return {"model_number": modelNum,
            "epochs": epochs,
            "train_batch_size": train_batch_size,
            "test_batch_size": test_batch_size,
            "learning_rate": lr,
            "gamma": def_gamma,
            "log_interval": def_log_interval}


def print_user_input(hyper_params):
    """
    print_user_input prints out all hyper parameters stored in dict
    Inputs:
        hyper_params: dict of hyper parameters in format of that returned by user_input()
    """

    # Print out model_num
    print("Model Number: {}".format(hyper_params["model_number"]))
    # Print out epochs
    print("Epochs: {}".format(hyper_params["epochs"]))
    # Print out training batch_size
    print("Training Batch Size: {}".format(hyper_params["train_batch_size"]))
    # Print out testing batch_size
    print("Testing Batch Size: {}".format(hyper_params["test_batch_size"]))
    # Print out learning_rate
    print("Learning Rate: {}".format(hyper_params["learning_rate"]))
    # Print out gamma
    print("Gamma: {}".format(hyper_params["gamma"]))
    # Print out log_interval
    print("Log Interval: {}".format(hyper_params["log_interval"]))


    return