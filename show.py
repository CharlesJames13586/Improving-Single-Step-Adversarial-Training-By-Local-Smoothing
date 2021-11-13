import argparse
import os
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--timestamp", default="2021-10-26", help="模型训练的时间戳，用来选择被测试的模型")

    return parser.parse_args()


def main():
    args = get_args()

    timestamp = args.timestamp
    logs_name = ""
    for filename in os.listdir("logs/"):
        if timestamp in filename:
            logs_name = "logs/" + filename
            break
    
    logs_file = open(logs_name)
    logs = logs_file.readlines()

    iter_record = []
    train_fgsm_acc = []
    train_pgd_acc = []
    test_fgsm_acc = []
    test_pgd_acc = []

    for line in logs:
        if not line[0].isdigit():
            continue

        words = re.split(",| ", line)
        iter_record.append(words[0][:-1])
        label = ""
        for i, word in enumerate(words):
            if word == "[train]":
                label = "train"
                continue
            if word == '[test]':
                label = "test"
                continue
            if word == "acc" and label == "train":
                train_fgsm_acc.append(float(words[i+1][:-1]))
            elif word == "acc_pgd-50-10" and label == "train":
                train_pgd_acc.append(float(words[i+1][:-1]))
            elif word == "acc_fgsm" and label == "test":
                test_fgsm_acc.append(float(words[i+1][:-1]))
            elif word == "acc_pgd-50-10" and label == "test":
                test_pgd_acc.append(float(words[i+1][:-1]))
    
    plt.plot(iter_record, train_fgsm_acc, label="[train] fgsm")
    plt.plot(iter_record, train_pgd_acc, label="[train] pgd-50-10")
    plt.plot(iter_record, test_fgsm_acc, label="[test] fgsm")
    plt.plot(iter_record, test_pgd_acc, label="[test] pgd-50-10")
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()