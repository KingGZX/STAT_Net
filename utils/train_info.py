import torch
import torch.optim as optim
import torch.nn as nn
from utils.train import vote
from loadata import Config
from utils.log import get_logger
from loadata import Dataset
import math
import numpy as np
from baselines.InfoGCN_baseline import get_mmd_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32


def batch_train_Info(model, dataset: Dataset, epochs: int, model_name: str, cfg: Config, items: list, log=True,
                     log_file=None, save=False, batchsize=4):
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

    batch_size = batchsize
    train_len = len(dataset.train_data)
    batches = math.ceil(train_len / batch_size)

    # move the model to GPU
    model.to(device)

    # create log file handler
    if log:
        import os
        if not os.path.exists("./log"):
            os.mkdir("./log")
        idx = len(os.listdir("./log")) + 1
        fp = "exp" + str(idx) + ".log" if log_file is None else log_file
        logger = get_logger(
            filename="./log/" + fp,
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

            out, z = model(train_data)
            loss = torch.tensor(0).to(device, dtype=dtype)

            i_loss = criterion(out, train_label[0])
            loss += i_loss
            correct_num += torch.sum(torch.argmax(out, dim=1) == train_label[0]).cpu()

            mmd_loss, l2_z_mean, z_mean = get_mmd_loss(z, model.z_prior, train_label[0], cfg.classes[items[0] - 1])
            loss = loss + 1e-4 * mmd_loss + 1e-1 * l2_z_mean

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


def validation(model, dataset: Dataset, items: list, logger):
    """
    :param model:     same as the train function above
    :param dataset:
    :param items:
    :return:
    """
    model.eval()
    with torch.no_grad():
        batches = len(dataset.test_data)
        total_score_acc = 0
        for batch in range(batches):
            test_data, test_labels, name, _ = dataset.load_data(train=False, items=items)
            # Note that, test data is a list of numpy arrays. We use majority vote to determine the final result
            predicted = list()
            for sub_cycle in test_data:
                sub_cycle = torch.tensor(sub_cycle, dtype=torch.float32).to(device, dtype=dtype)

                sub_out, z = model(sub_cycle)

                sub_predicted = list()
                predict_i = torch.argmax(sub_out, dim=1).cpu().item()
                sub_predicted.append(predict_i)
                predicted.append(sub_predicted)

            predicted = vote(predicted)
            for i in range(len(items)):
                if predicted[i] != test_labels[i]:
                    logger.info("patient {}: item {}, true label is {}, but predict {}".format(name, items[i],
                                                                                               test_labels[i],
                                                                                               predicted[i]))

            true_total_scores = np.sum(test_labels)
            pred_total_scores = np.sum(predicted)
            logger.info("patient {}:  true total score is {}, predict {}".format(name, true_total_scores,
                                                                                 pred_total_scores))
            if true_total_scores == pred_total_scores:
                total_score_acc += 1

        total_correct_num = total_score_acc
        total_score_acc /= batches
        logger.info("total score accuracy is {}%".format(total_score_acc * 100))
        return total_score_acc, total_correct_num