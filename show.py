import argparse
import os
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--timestamp", default="153624", help="模型训练的时间戳，用来选择被测试的模型")

    return parser.parse_args()


def main(timestamp):
    # args = get_args()

    # timestamp = args.timestamp
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
    test_acc = []
    test_fgsm_acc = []
    test_pgd_acc = []
    g_i_s = []

    for line in logs:
        if not line[0].isdigit():
            continue

        words = re.split(",| ", line)
        iter_record.append(int(words[0].split('-')[0]))
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
            elif word == "acc_pgd-10-1" and label == "train":
                train_pgd_acc.append(float(words[i+1][:-1]))
            elif word == "acc_clean" and label == "test":
                test_acc.append(float(words[i+1][:-1]))
            elif word == "acc_fgsm" and label == "test":
                test_fgsm_acc.append(float(words[i+1][:-1]))
            elif word == "acc_pgd-50-10" and label == "test":
                test_pgd_acc.append(float(words[i+1][:-1]))
            elif word == "acc_pgd-10-1" and label == "test":
                test_pgd_acc.append(float(words[i+1][:-1]))
            elif word == "g_i_s":
                g_i_s.append(float(words[i+1]))
    
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(111)
    ax1.plot(iter_record, train_fgsm_acc, label="[train] FGSM")
    # plt.plot(iter_record, train_pgd_acc, label="[train] PGD-50-10")
    ax1.plot(iter_record, train_pgd_acc, label="[train] PGD-10")
    ax1.plot(iter_record, test_acc, label="[test] clean")
    ax1.plot(iter_record, test_fgsm_acc, label="[test] FGSM")
    # plt.plot(iter_record, test_pgd_acc, label="[test] PGD-50-10")
    ax1.plot(iter_record, test_pgd_acc, label="[test] PGD-10")
    # plt.plot(iter_record, test_pgd_acc, label=timestamp + " pgd-50-10")
    # ax1.gca().xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax1.set_ylabel("accuracy")
    ax1.legend(loc='upper left')
    # plt.legend()
    ax2 = ax1.twinx()
    ax2.plot(iter_record, g_i_s, '--', label="$\Omega$", color='indigo')
    ax2.legend(loc=(0.85, 0.85))
    plt.xlabel("epoch")
    plt.xlim(left=-1, right=31)
    # x_labels = ['']*31
    # x_labels[0] = 1
    # x_labels[9] = 10
    # x_labels[19] = 20
    # x_labels[29] = 30
    # plt.xticks(ticks=range(31), labels=x_labels)
    
    # plt.legend(fontsize=10)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))

    
    plt.savefig("test.pdf",format="pdf")
    # plt.show()

if __name__ == "__main__":
    # main("153624")
    main("211654")
    
    # plt.savefig("test", dpi=400)
    
    # plt.show()