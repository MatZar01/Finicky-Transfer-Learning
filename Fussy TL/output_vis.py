import os
import sys
import csv
import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt


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

    ax = plt.axes()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()

    plt.ylabel("Model accuracy [%]")
    plt.xlabel("Parameters left in model [%]")
    plt.title("Pruned model accuracy")
    plt.gcf().canvas.set_window_title("Pruned model accuracy")
    plt.grid(color='grey', linestyle=':')

    plot_acc, = plt.plot(p_left, acc)
    plot_b_acc, = plt.plot([min(p_left), max(p_left)], [acc[-1], acc[-1]], 'r')
    plt.legend([plot_acc, plot_b_acc], ['Pruned models', 'Base model'], loc='lower center', frameon=False)
    plt.show()


def plot_pruned_size_and_time(data_dict):
    p_left = get_property_list(data_dict["models"], "parameters left")
    p_left = list(np.float_(p_left))

    s = get_property_list(data_dict["models"], "size")
    s = list(np.float_(s))

    t = get_property_list(data_dict["models"], "time")
    t = list(np.float_(t))

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Parameters left in model [%]")
    ax1.set_ylabel("Size [MB]")
    ax1.plot(p_left, s)
    ax1.grid(color='grey', linestyle=':')
    ax1.spines["top"].set_visible(False)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Time [s]")
    ax2.plot(p_left, t, 'r')
    ax2.spines["top"].set_visible(False)

    plt.title("Pruned model size and inference time")
    plt.gcf().canvas.set_window_title("Pruned model size and inference time")
    s_patch = mlines.Line2D([],[],color='b', label='Size')
    t_patch = mlines.Line2D([],[],color='r', label='Time')
    plt.legend(handles=[s_patch, t_patch], loc='lower center', frameon=False)
    plt.show()


def plot_pruned_parameters_to_layers(data_dict):
    p_left = get_property_list(data_dict["models"], "parameters left")
    p_left = list(np.float_(p_left))

    l_left = get_property_list(data_dict["models"], "layers left")
    l_left = list(np.float_(l_left) * 100)

    ax = plt.axes()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()

    plt.ylabel("Parameters left in model [%]")
    plt.xlabel("Layers left in model [%]")
    plt.title("Layers to parameters")
    plt.gcf().canvas.set_window_title("Layers to parameters")
    plt.grid(color='grey', linestyle=':')
    plt.plot(l_left, p_left)
    plt.show()


def plot_pruned_decision_f(data_dict):
    p_left = get_property_list(data_dict["models"], "parameters left")
    p_left = p_left[:len(p_left) - 1]
    p_left = list(np.float_(p_left))

    s = get_property_list(data_dict["models"], "score")
    s = s[:len(s) - 1]
    s = list(np.float_(s))

    ax = plt.axes()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()

    plt.ylabel("Score")
    plt.xlabel("Parameters left in model [%]")
    plt.title("Value of decision function")
    plt.grid(color='grey', linestyle=':')
    plt.plot(p_left, s)
    plt.show()


def print_help():
    print("Avaiable commands:")
    print("--help to display this message again")
    print("--csv_file_path to specify output .cvs file path")


def main(csv_file_path):
    if csv_file_path is "--help":
        print_help()
        return

    if os.path.exists(csv_file_path):
        data_dict = read_csv_data(csv_file_path)

        plot_pruned_model_acc(data_dict)
        plot_pruned_size_and_time(data_dict)
        plot_pruned_parameters_to_layers(data_dict)
        plot_pruned_decision_f(data_dict)
    else:
        print("File do not exist")
        print_help()


if __name__ == "__main__":
    print(sys.argv[0])
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Specify .csv path")
        print_help()
