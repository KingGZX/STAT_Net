from train import *
from loadata import Config
from utils.log import get_logger
from loadata import Dataset
import math
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32


def batch_train_CL(model, dataset: Dataset, epochs: int, model_name: str, cfg: Config, items: list, log=True,
                log_file=None, save=False):
    """
    :param model:        Network Instance
    :param dataset:      The class implemented in loadata.py
    :param epochs:       Hyperparameter
    :param model_name:   e.g. ST-GCN, Uniformer, GFN, for plotting result
    :param cfg:          config file
    :param items:        item needs to be trained by this model
    :param log_file:     Logging file name
    :param log:          Logging setting
    :param save:         whether to save the pth format of model
    :return:
    """
    # Hyperparameter setting
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    batch_size = cfg.batch_size
    train_len = len(dataset.train_data)
    batches = math.ceil(train_len / batch_size)

    # move the model to GPU
    model.to(device)

    # create log file handler
    if log:
        import os
        if not os.path.exists("../log"):
            os.mkdir("../log")
        idx = len(os.listdir("../log")) + 1
        fp = "exp" + str(idx) + ".log" if log_file is None else log_file
        logger = get_logger(
            filename="../log/" + fp,
            verbosity=1
        )

    # for single item
    max_acc = 0
    max_num = 0
    
    for epoch in range(epochs):
        # shuffle gait cycles in the train set
        logger.info("Epoch: {}/{}".format(epoch + 1, epochs))
        dataset.shuffle()
        total_loss = 0
        correct_num = 0
        for batch in range(batches):
            if batch_size > 1:
                train_data, train_label = dataset.load_batch_data_train(items=items)
            else:
                train_data, train_label, patient_name, gait_cycle = dataset.load_data(train=True, items=items)
            # to tensor
            train_data = torch.tensor(train_data, dtype=torch.float32).to(device, dtype=dtype)
            
            for i in range(len(items)):
                # each element of train_label is a list, it's corresponding labels of this batch data of items[i]
                train_label[i] = torch.tensor(train_label[i], dtype=torch.long).to(device)

            out, cl_loss = model(train_data, train_label[0], get_cl_loss=True)
            loss = torch.tensor(0).to(device, dtype=dtype)
            loss += cl_loss

            i_loss = criterion(out, train_label[0])
            loss += i_loss
            correct_num += torch.sum(torch.argmax(out, dim=1) == train_label[0]).cpu()

            total_loss += loss.item()

            # backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch % 100 == 0:
                total_score_acc, total_correct_num = validation(model, dataset, items, logger)
                print("Epoch:{}/{}, Test Set | average accuracy is {}%".format(epoch + 1, epochs, total_score_acc * 100))
                model.train()
                if max_acc < total_score_acc:
                    max_acc = total_score_acc   
                    max_num = total_correct_num 

        total_loss = total_loss / batches
        train_accuracy = correct_num / train_len

        total_score_acc, total_correct_num = validation(model, dataset, items, logger)
        print("Epoch:{}/{}, Test Set | average accuracy is {}%".format(epoch + 1, epochs, total_score_acc * 100))
        model.train()
        if max_acc < total_score_acc:
            max_acc = total_score_acc
            max_num = total_correct_num 
            
        print("Epoch:{}/{}, Train Set | average loss is {} | average accuracy is {}%".format(epoch + 1, epochs, total_loss, train_accuracy * 100))
        
    logger.info("Model {}, Maximum Accuracy is {}%".format(model_name, max_acc * 100))
    
    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()

    return max_num