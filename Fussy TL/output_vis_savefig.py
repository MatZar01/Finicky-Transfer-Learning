import os
import sys
import csv
import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt




import matplotlib as mpl
import matplotlib.figure as figure
import matplotlib.markers as markers
mpl.rc('text', usetex = True)
mpl.rc('font', size = 10)


def create_model_entry(model_data):
    return {"name": model_data[0],
            "layers left": model_data[1],
            "parameters": model_data[2],
            "parameters left": model_data[3],
            "accuracy": model_data[4],
            "Ref accuracy": model_data[5],
            "f1": model_data[6],
            "size": model_data[7],
            "time": model_data[9],
            "score": model_data[12]}


def add_model_entry(data_dict, data):
    data_dict["models"].append(create_model_entry(data))


def create_empty_data_dict():
    return {"models": []}


def read_csv_data(file_path):
    csv_data = create_empty_data_dict()
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        i = 0
        for row in csv_reader:
            if i == 0:
                i += 1
                continue

            add_model_entry(csv_data, row)
            i += 1
    return csv_data


def get_property_list(data_dict, p_name):
    p_list = []
    for data in data_dict:
        p_list.append(data[p_name])
    return p_list


def get_data_range(data, base):
    return [base * round(min(data) / base), base * round(max(data) / base)]


def plot_pruned_model_acc(data_dict):
    p_left = get_property_list(data_dict["models"], "parameters left")
    p_left = list(np.float_(p_left))

    acc = get_property_list(data_dict["models"], "accuracy")
    acc = list(np.float_(acc))

    fig = mpl.figure.Figure(figsize=(4,3))
    ax = fig.add_subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()

    ax.set_ylabel("model accuracy [\%]")
    ax.set_xlabel("parameters left in the model [\%]")
    ax.set_title("Pruned model accuracy")
    ax.grid(alpha=0.75, ls=':',color='grey')

    ax.plot([min(p_left), max(p_left)], [acc[-1], acc[-1]], 'r', label='Base model')
    ax.plot(p_left, label='Pruned models')
    fig.tight_layout()
    fig.savefig("plot_pruned_model_acc.pdf")


def plot_pruned_size_and_time(data_dict):
    p_left = get_property_list(data_dict["models"], "parameters left")
    p_left = list(np.float_(p_left))

    s = get_property_list(data_dict["models"], "size")
    s = list(np.float_(s))

    t = get_property_list(data_dict["models"], "time")
    t = list(np.float_(t))

    fig = mpl.figure.Figure(figsize=(4,3))
    ax1 = fig.add_subplot(111)

    ax1.set_xlabel(r"parameters left in the model [\%]")
    ax1.set_ylabel("size [MB]")
    ax1.set_title("Pruned model size and inference time")
    
    ax1.plot(p_left, s, "--",label="size")
    ax1.grid(color='grey', linestyle=':')
    ax1.spines["top"].set_visible(False)

    ax2 = ax1.twinx()
    ax2.set_ylabel("time [s]")
    ax2.plot(p_left, t, 'r:', label="time")
    ax2.spines["top"].set_visible(False)

    
    fig.legend(loc='center', ncol=2, bbox_to_anchor=(0.3, 0., 0.5, 0.5))
    fig.tight_layout()
    fig.savefig("plot_pruned_size_and_time.pdf")
    


def plot_pruned_parameters_to_layers(data_dict):
    p_left = get_property_list(data_dict["models"], "parameters left")
    p_left = list(np.float_(p_left))

    l_left = get_property_list(data_dict["models"], "layers left")
    l_left = list(np.float_(l_left) * 100)

    fig = mpl.figure.Figure(figsize=(4,3))
    ax = fig.add_subplot(111)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()

    ax.set_ylabel("parameters left in the model [\%]")
    ax.set_xlabel("layers left in the model [\%]")
    ax.set_title("Relation between layers and parameters")
    ax.grid(alpha=0.75, ls=':',color='grey')
    ax.plot(l_left[0:-1:5], p_left[0:-1:5], "bd:", lw=1, fillstyle='none')
    
    fig.tight_layout()
    fig.savefig("plot_pruned_parameters_to_layers.pdf")


def plot_pruned_decision_f(data_dict):
    p_left = get_property_list(data_dict["models"], "parameters left")
    p_left = p_left[:len(p_left) - 1]
    p_left = list(np.float_(p_left))

    s = get_property_list(data_dict["models"], "score")
    s = s[:len(s) - 1]
    s = list(np.float_(s))

    fig = mpl.figure.Figure(figsize=(4,3))
    ax = fig.add_subplot(111)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()

    ax.set_ylabel("score")
    ax.set_xlabel("parameters left in the model [\%]")
    ax.set_title("Value of the decision function")
    ax.grid(alpha=0.75, ls=':',color='grey')
    ax.plot(p_left[0:-1:5], s[0:-1:5], "bv:", lw=1, fillstyle='none')

    fig.tight_layout()
    fig.savefig("plot_pruned_decision_f.pdf")


def print_help():
    print("Avaiable commands:")
    print("--help to display this message again")
    print("--csv_file_path to specify output .cvs file path")


def main(csv_file_path):
    csv_file_path = csv_file_path.replace('-', '')
    if csv_file_path == "help":
        print_help()
        return

    if os.path.exists(csv_file_path):
        data_dict = read_csv_data(csv_file_path)

        plot_pruned_model_acc(data_dict)
        plot_pruned_size_and_time(data_dict)
        plot_pruned_parameters_to_layers(data_dict)
        plot_pruned_decision_f(data_dict)
    else:
        print("File does not exist")
        print_help()


if __name__ == "__main__":
    print(sys.argv[0])
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Specify .csv path")
        print_help()
