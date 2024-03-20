import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from utils.plot import *
from loadata import Config
from utils.log import get_logger
from loadata import Dataset
import math
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32


def draw(train_acc, test_acc, item_loss, total_loss, total_score_acc, model: str, cfg: Config, items, epochs):
    # draw train accuracy, test accuracy and loss on the same figure
    import os
    if not os.path.exists("../Results"):
        os.mkdir("../Results")
    idx = len(os.listdir("../Results")) + 1
    fp = "./Results/exp" + str(idx)
    os.mkdir(fp)
    item_len = len(items)
    items_test_acc_list = list()
    item_names = list()
    for item in range(item_len):
        item_train, item_test, item_los = list(), list(), list()
        for epoch in range(epochs):
            item_train.append(train_acc[epoch][item])
            item_test.append(test_acc[epoch][item])
            item_los.append(item_loss[epoch][item])
        data_list = [item_train, item_test, item_los]
        legend_name = ["train accuracy", "test accuracy", "loss value"]
        item_name = cfg.label_dict["item" + str(items[item])]
        items_test_acc_list.append(item_test)
        item_names.append(item_name)
        title = model + "'s performance on item " + item_name
        item_acc_loss_draw(epochs, data_list=data_list, legend_name=legend_name,
                           title_name=title, save_path=fp, item_name="item" + str(items[item]))

    # draw a single picture of total loss trend
    if item_len > 1:
        total_loss_draw(epochs, loss_list=total_loss, legend_name="total loss",
                        title_name="total loss of model " + model, save_path=fp, name="total_loss")

    # draw a single picture of total score accuracy trend
    if item_len > 1:
        total_loss_draw(epochs, loss_list=total_score_acc, legend_name="total score accuracy",
                        title_name="total score accuracy of model " + model, save_path=fp, name="total_score_accuracy")

    # make comparison of different items
    if item_len > 1:
        compare_item_acc_draw(epochs, items_test_acc_list, item_names, "test accuracy of each item", fp, "compare")


def batch_train_stat(model, dataset: Dataset, epochs: int, model_name: str, cfg: Config,
                     items: list, log=True, log_file=None):
    """
    :param dis_model:    distance relevant features model
    :param ang_model:
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

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    criterion = nn.CrossEntropyLoss()

    batch_size = cfg.batch_size
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

        model1_acc = 0
        model2_acc = 0

        for batch in range(batches):
            if batch_size > 1:
                train_data, train_label = dataset.load_batch_data_train(items=items, batchsize=cfg.batch_size)
            else:
                train_data, train_label, patient_name, gait_cycle = dataset.load_data(train=True, items=items)
            # to tensor
            train_data = torch.tensor(train_data, dtype=torch.float32).to(device, dtype=dtype)

            for i in range(len(items)):
                # each element of train_label is a list, it's corresponding labels of this batch data of items[i]
                train_label[i] = torch.tensor(train_label[i], dtype=torch.long).to(device)

            dis_out, ang_out, cl_loss = model(train_data, train_label[0], True)

            loss = torch.tensor(0).to(device, dtype=dtype)
            loss = loss + cl_loss

            i_loss1 = criterion(dis_out, train_label[0])
            i_loss2 = criterion(ang_out, train_label[0])

            loss = loss + i_loss1 + i_loss2

            model1_acc += torch.sum(torch.argmax(dis_out, dim=1) == train_label[0]).cpu().item()
            model2_acc += torch.sum(torch.argmax(ang_out, dim=1) == train_label[0]).cpu().item()

            total_loss += loss.item()

            # backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch % 100 == 0:
                total_score_acc, total_correct_num = validation(model, dataset, items, logger, model_name)
                print("Epoch:{}/{}, Test Set | average accuracy is {}%".format(epoch + 1, epochs, total_score_acc * 100))
                model.train()
                if max_acc < total_score_acc:
                    max_acc = total_score_acc   
                    max_num = total_correct_num 

        scheduler.step()
        model1_acc = model1_acc * 100 / train_len
        model2_acc = model2_acc * 100 / train_len
        total_loss = total_loss / batches

        total_score_acc, total_correct_num = validation(model, dataset, items, logger, model_name)
        print("Epoch:{}/{}, Test Set | average accuracy is {}%".format(epoch + 1, epochs, total_score_acc * 100))
        model.train()
        if max_acc < total_score_acc:
            max_acc = total_score_acc
            max_num = total_correct_num 
        
        print("Epoch:{}/{} | Train Set | Average training loss is {}, \
              Average accuracy of dis_model is {}% | Average accuracy of ang_model is {}%."
              .format(epoch + 1, epochs, total_loss, model1_acc, model2_acc))

    logger.info("Maximum Accuracy is {}".format(max_acc))

    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()

    return max_num


def vote(scores: list):
    """
    :param scores:      prediction of each gait cycle of the same person
    :return:
            choose the majority as the label
            if there is not only 1 mode, then use average as the final score.
    """
    new_scores = list()
    sub_cycles = len(scores)  # number of gait cycles
    items = len(scores[0])  # number of items
    for i in range(items):
        item_i_prediction = list()
        for j in range(sub_cycles):
            item_i_prediction.append(scores[j][i])
        # find the mode
        d = dict()
        for pred in item_i_prediction:
            d[pred] = d[pred] + 1 if pred in d else 1
        i_max = 0
        prediction = list()
        for key in d:
            # d[key] is the occurrence of score "key"
            if d[key] > i_max:
                i_max = d[key]
                prediction.clear()
                prediction.append(key)
            elif d[key] == i_max:
                prediction.append(key)
        new_scores.append(round(sum(prediction) / len(prediction)))

    return new_scores


def validation(model, dataset: Dataset, items: list, logger, model_name):
    """
    :param dis_model:    distance relevant features model
    :param ang_model:
    :param dataset:
    :param items:
    :return:
    """
    model.eval()
    with torch.no_grad():
        batches = len(dataset.test_data)
        item_acc_list = np.zeros(len(items))
        total_score_acc = 0
        for batch in range(batches):
            test_data, test_labels, name, _ = dataset.load_data(train=False, items=items)
            # Note that, test data is a list of numpy arrays. We use majority vote to determine the final result
            predicted = list()
            for index, sub_cycle in enumerate(test_data):
                sub_cycle = torch.tensor(sub_cycle, dtype=torch.float32).to(device, dtype=dtype)

                sub_dis_out, sub_ang_out = model(sub_cycle)

                sub_predicted = list()
                prob1 = F.softmax(sub_dis_out, dim=1)
                prob2 = F.softmax(sub_ang_out, dim=1)
                predict_i = torch.argmax(prob1 + prob2, dim=1).cpu().item()

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
        logger.info("model {}: total score accuracy is {}%".format(model_name, total_score_acc * 100))
        return total_score_acc, total_correct_num
