#!/usr/bin/python
from gurobipy import *
import pandas as pd
import sys
import time
from Tree import Tree
from BendersOCT import BendersOCT
import logger
import getopt
import csv
from sklearn.model_selection import train_test_split
from utils import *
from logger import logger


def get_left_exp_integer(master, b, n, i):
    lhs = quicksum(-master.m[i] * master.b[n, f] for f in master.cat_features if master.data.at[i, f] == 0)

    return lhs


def get_right_exp_integer(master, b, n, i):
    lhs = quicksum(-master.m[i] * master.b[n, f] for f in master.cat_features if master.data.at[i, f] == 1)

    return lhs


def get_target_exp_integer(master, p, beta, n, i):
    label_i = master.data.at[i, master.label]

    if master.mode == "classification":
        lhs = -1 * master.beta[n, label_i]
    elif master.mode == "regression":
        # min (m[i]*p[n] - y[i]*p[n] + beta[n] , m[i]*p[n] + y[i]*p[n] - beta[n])
        if master.m[i] * p[n] - label_i * p[n] + beta[n, 1] < master.m[i] * p[n] + label_i * p[n] - beta[n, 1]:
            lhs = -1 * (master.m[i] * master.p[n] - label_i * master.p[n] + master.beta[n, 1])
        else:
            lhs = -1 * (master.m[i] * master.p[n] + label_i * master.p[n] - master.beta[n, 1])

    return lhs


def get_cut_integer(master, b, p, beta, left, right, target, i):
    lhs = LinExpr(0) + master.g[i]
    for n in left:
        tmp_lhs = get_left_exp_integer(master, b, n, i)
        lhs = lhs + tmp_lhs

    for n in right:
        tmp_lhs = get_right_exp_integer(master, b, n, i)
        lhs = lhs + tmp_lhs

    for n in target:
        tmp_lhs = get_target_exp_integer(master, p, beta, n, i)
        lhs = lhs + tmp_lhs

    return lhs


def subproblem(master, b, p, beta, i):
    label_i = master.data.at[i, master.label]
    current = 1
    right = []
    left = []
    target = []
    subproblem_value = 0

    while True:
        pruned, branching, selected_feature, terminal, current_value = get_node_status(master, b, beta, p, current)
        if terminal:
            target.append(current)
            if current in master.tree.Nodes:
                left.append(current)
                right.append(current)
            if master.mode == "regression":
                subproblem_value = master.m[i] - abs(current_value - label_i)
            elif master.mode == "classification" and beta[current, label_i] > 0.5:
                subproblem_value = 1
            break
        elif branching:
            if master.data.at[i, selected_feature] == 1:  # going right on the branch
                left.append(current)
                target.append(current)
                current = master.tree.get_right_children(current)
            else:  # going left on the branch
                right.append(current)
                target.append(current)
                current = master.tree.get_left_children(current)

    return subproblem_value, left, right, target


##########################################################
# Defining the callback function
###########################################################
def mycallback(model, where):
    '''
    This function is called by gurobi at every node through the branch-&-bound tree while we solve the model.
    Using the argument "where" we can see where the callback has been called. We are specifically interested at nodes
    where we get an integer solution for the master problem.
    When we get an integer solution for b and p, for every datapoint we solve the subproblem which is a minimum cut and
    check if g[i] <= value of subproblem[i]. If this is violated we add the corresponding benders constraint as lazy
    constraint to the master problem and proceed. Whenever we have no violated constraint! It means that we have found
    the optimal solution.
    :param model: the gurobi model we are solving.
    :param where: the node where the callback function is called from
    :return:
    '''
    data_train = model._master.data
    mode = model._master.mode

    local_eps = 0.0001
    if where == GRB.Callback.MIPSOL:
        func_start_time = time.time()
        model._callback_counter_integer += 1
        # we need the value of b,w and g
        g = model.cbGetSolution(model._vars_g)
        b = model.cbGetSolution(model._vars_b)
        p = model.cbGetSolution(model._vars_p)
        beta = model.cbGetSolution(model._vars_beta)

        added_cut = 0
        # We only want indices that g_i is one!
        for i in data_train.index:
            if mode == "classification":
                g_threshold = 0.5
            elif mode == "regression":
                g_threshold = 0
            if g[i] > g_threshold:
                subproblem_value, left, right, target = subproblem(model._master, b, p, beta, i)
                if mode == "classification" and subproblem_value == 0:
                    added_cut = 1
                    lhs = get_cut_integer(model._master, b, p, beta, left, right, target, i)
                    model.cbLazy(lhs <= 0)
                elif mode == "regression" and ((subproblem_value + local_eps) < g[i]):
                    added_cut = 1
                    lhs = get_cut_integer(model._master, b, p, beta, left, right, target, i)
                    model.cbLazy(lhs <= 0)

        func_end_time = time.time()
        func_time = func_end_time - func_start_time
        # print(model._callback_counter)
        model._total_callback_time_integer += func_time
        if added_cut == 1:
            model._callback_counter_integer_success += 1
            model._total_callback_time_integer_success += func_time


def main(argv):
    print(argv)
    input_file = None
    depth = None
    time_limit = None
    _lambda = None
    input_sample = None
    calibration = None
    mode = "classification"
    '''
    Depending on the value of input_sample we choose one of the following random seeds and then split the whole data
    into train, test and calibration
    '''
    random_states_list = [41, 23, 45, 36, 19, 123]

    try:
        opts, args = getopt.getopt(argv, "f:d:t:l:i:c:m:",
                                   ["input_file=", "depth=", "timelimit=", "lambda=",
                                    "input_sample=",
                                    "calibration=", "mode="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-f", "--input_file"):
            input_file = arg
        elif opt in ("-d", "--depth"):
            depth = int(arg)
        elif opt in ("-t", "--timelimit"):
            time_limit = int(arg)
        elif opt in ("-l", "--lambda"):
            _lambda = float(arg)
        elif opt in ("-i", "--input_sample"):
            input_sample = int(arg)
        elif opt in ("-c", "--calibration"):
            calibration = int(arg)
        elif opt in ("-m", "--mode"):
            mode = arg

    start_time = time.time()
    data_path = os.getcwd() + '/../../DataSets/'
    data = pd.read_csv(data_path + input_file)
    '''Name of the column in the dataset representing the class label.
    In the datasets we have, we assume the label is target. Please change this value at your need'''
    label = 'target'

    # Tree structure: We create a tree object of depth d
    tree = Tree(depth)

    ##########################################################
    # output setup
    ##########################################################
    approach_name = 'FlowOCT'
    out_put_name = input_file + '_' + str(input_sample) + '_' + approach_name + '_d_' + str(depth) + '_t_' + str(
        time_limit) + '_lambda_' + str(
        _lambda) + '_c_' + str(calibration)
    out_put_path = os.getcwd() + '/../../Results/'
    # Using logger we log the output of the console in a text file
    sys.stdout = logger(out_put_path + out_put_name + '.txt')

    ##########################################################
    # data splitting
    ##########################################################
    '''
    Creating  train, test and calibration datasets
    We take 50% of the whole data as training, 25% as test and 25% as calibration

    When we want to calibrate _lambda, for a given value of _lambda we train the model on train and evaluate
    the accuracy on calibration set and at the end we pick the _lambda with the highest accuracy.

    When we got the calibrated _lambda, we train the mode on (train+calibration) which we refer to it as
    data_train_calibration and evaluate the accuracy on (test)

    '''
    data_train, data_test = train_test_split(data, test_size=0.25, random_state=random_states_list[input_sample - 1])
    data_train_calibration, data_calibration = train_test_split(data_train, test_size=0.33,
                                                                random_state=random_states_list[input_sample - 1])

    if calibration == 1:  # in this mode, we train on 50% of the data; otherwise we train on 75% of the data
        data_train = data_train_calibration

    train_len = len(data_train.index)
    ##########################################################
    # Creating and Solving the problem
    ##########################################################
    # We create the master problem by passing the required arguments
    master = BendersOCT(data_train, label, tree, _lambda, time_limit, mode)

    master.create_master_problem()
    master.model.update()
    master.model.optimize(mycallback)
    end_time = time.time()
    solving_time = end_time - start_time

    ##########################################################
    # Preparing the output
    ##########################################################
    b_value = master.model.getAttr("X", master.b)
    beta_value = master.model.getAttr("X", master.beta)
    p_value = master.model.getAttr("X", master.p)

    print("\n\n")
    print_tree(master, b_value, beta_value, p_value)

    print('\n\nTotal Solving Time', solving_time)
    print("obj value", master.model.getAttr("ObjVal"))

    print('Total Callback counter (Integer)', master.model._callback_counter_integer)
    print('Total Successful Callback counter (Integer)', master.model._callback_counter_integer_success)

    print('Total Callback Time (Integer)', master.model._total_callback_time_integer)
    print('Total Successful Callback Time (Integer)', master.model._total_callback_time_integer_success)

    # print(b_value)
    # print(p_value)
    # print(beta_value)
    ##########################################################
    # Evaluation
    ##########################################################
    '''
    For classification we report accuracy
    For regression we report MAE (Mean Absolute Error) , MSE (Mean Squared Error) and  R-squared

    over training, test and the calibration set
    '''
    train_acc = test_acc = calibration_acc = 0
    train_mae = test_mae = calibration_mae = 0
    train_r_squared = test_r_squared = calibration_r_squared = 0

    if mode == "classification":
        train_acc = get_acc(master, data_train, b_value, beta_value, p_value)
        test_acc = get_acc(master, data_test, b_value, beta_value, p_value)
        calibration_acc = get_acc(master, data_calibration, b_value, beta_value, p_value)
    elif mode == "regression":
        train_mae = get_mae(master, data_train, b_value, beta_value, p_value)
        test_mae = get_mae(master, data_test, b_value, beta_value, p_value)
        calibration_mae = get_mae(master, data_calibration, b_value, beta_value, p_value)

        train_mse = get_mse(master, data_train, b_value, beta_value, p_value)
        test_mse = get_mse(master, data_test, b_value, beta_value, p_value)
        calibration_mse = get_mse(master, data_calibration, b_value, beta_value, p_value)

        train_r2 = get_r_squared(master, data_train, b_value, beta_value, p_value)
        test_r2 = get_r_squared(master, data_test, b_value, beta_value, p_value)
        calibration_r2 = get_r_squared(master, data_calibration, b_value, beta_value, p_value)

    print("obj value", master.model.getAttr("ObjVal"))
    if mode == "classification":
        print("train acc", train_acc)
        print("test acc", test_acc)
        print("calibration acc", calibration_acc)
    elif mode == "regression":
        print("train mae", train_mae)
        print("train mse", train_mse)
        print("train r^2", train_r_squared)

    ##########################################################
    # writing info to the file
    ##########################################################
    master.model.write(out_put_path + out_put_name + '.lp')
    # writing info to the file
    result_file = out_put_name + '.csv'
    with open(out_put_path + result_file, mode='a') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        if mode == "classification":
            results_writer.writerow(
                [approach_name, input_file, train_len, depth, _lambda, time_limit,
                 master.model.getAttr("Status"), master.model.getAttr("ObjVal"), train_acc,
                 master.model.getAttr("MIPGap") * 100, master.model.getAttr("NodeCount"), solving_time,
                 master.model._total_callback_time_integer, master.model._total_callback_time_integer_success,
                 master.model._callback_counter_integer, master.model._callback_counter_integer_success,
                 test_acc, calibration_acc, input_sample])
        elif mode == "regression":
            results_writer.writerow(
                [approach_name, input_file, train_len, depth, _lambda, time_limit,
                 master.model.getAttr("Status"),
                 master.model.getAttr("ObjVal"), train_mae, train_mse, train_r_squared,
                 master.model.getAttr("MIPGap") * 100, master.model.getAttr("NodeCount"), solving_time,
                 master.model._total_callback_time_integer, master.model._total_callback_time_integer_success,
                 master.model._callback_counter_integer, master.model._callback_counter_integer_success,
                 test_mae, calibration_mae,
                 test_mse, calibration_mse,
                 test_r_squared, calibration_r2,
                 input_sample])


if __name__ == "__main__":
    main(sys.argv[1:])
