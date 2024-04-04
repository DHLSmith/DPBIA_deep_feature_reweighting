"""Evaluate DFR on spurious correlations datasets."""

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import tqdm
import argparse
import sys
from collections import defaultdict
import json
from functools import partial
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from wb_data import WaterBirdsDataset, get_loader, get_transform_cub, log_data
from utils import Logger, AverageMeter, set_seed, evaluate, get_y_p

from datetime import datetime

if __name__ == '__main__':
    # WaterBirds
    C_OPTIONS = [1., 0.7, 0.3, 0.1, 0.07, 0.03, 0.01]
    CLASS_WEIGHT_OPTIONS = [1., 2., 3., 10., 100., 300., 1000.]
    # CelebA
    REG = "l1"
    # # REG = "l2"
    # C_OPTIONS = [3., 1., 0.3, 0.1, 0.03, 0.01, 0.003]
    # CLASS_WEIGHT_OPTIONS = [1., 2., 3., 10., 100, 300, 500]

    CLASS_WEIGHT_OPTIONS = [{0: 1, 1: w} for w in CLASS_WEIGHT_OPTIONS] + [
            {0: w, 1: 1} for w in CLASS_WEIGHT_OPTIONS]


    parser = argparse.ArgumentParser(description="Tune and evaluate DFR.")
    parser.add_argument(
        "--data_dir", type=str,
        default=None,
        help="Train dataset directory")
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="Checkpoint path")
    parser.add_argument(
        "--batch_size", type=int, default=100, required=False,
        help="Checkpoint path")
    parser.add_argument(
        "--balance_dfr_val", type=bool, default=True, required=False,
        help="Subset validation to have equal groups for DFR(Val)")
    parser.add_argument(
        "--notrain_dfr_val", type=bool, default=True, required=False,
        help="Do not add train data for DFR(Val)")
    parser.add_argument(
        "--tune_class_weights_dfr_train", action='store_true',
        help="Learn class weights for DFR(Train)")
    parser.add_argument(
        "--output_dir", type=str,
        default="logs",
        help="Output directory")
    args = parser.parse_args()

    date_time = datetime.now().strftime("_%Y-%m-%d_%H%M%S")
    args.output_dir += date_time
    print('Preparing directory %s' % args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    logger = Logger(os.path.join(args.output_dir, 'dfr_log.txt'))

    def dfr_on_validation_tune(
            all_embeddings, all_y, all_g, preprocess=True,
            balance_val=False, add_train=True, num_retrains=1):
            #Ajayesh adds- 
            #this do hyperparameters tunning here we tune strength of the regularization(lambda or c) 
            #and also class weights in case of we also use training data

            #input:- all_embeddings:embeddings extracted from feature extracter of resnet50 bascically input to last layer
            #        all_y:class labels for ex 0 for landbird and 1 for waterbird
            #        all_g: group label ex landbird on land or waterbird on land and so on(calculated in wb_data.py)
            #        preprocess: flag for weather to standardize embedding to have zero mean and unit standard deviation
            #        balance_val: flag for weather to balance validation data ie to take all of the minority group and take the same size
            #        from other groups and combining them.
            #        add_train: flag for weather to use training data as well (we also tune for class weights in this case)
            #        num_retrain: how many time we retrain last layer
            #output:- best_hypers: the hyperpameters that gave best accuracy on worst group [c(strenght of regularization term),weightclass1, weightclass2]

            
        worst_accs = {}
        for i in range(num_retrains):
            x_val = all_embeddings["val"]
            y_val = all_y["val"]
            g_val = all_g["val"]
            n_groups = np.max(g_val) + 1
            # taking one half of validation data for hyperparameter tuning
            n_val = len(x_val) // 2
            idx = np.arange(len(x_val))
            np.random.shuffle(idx)

            x_valtrain = x_val[idx[n_val:]]
            y_valtrain = y_val[idx[n_val:]]
            g_valtrain = g_val[idx[n_val:]]
            #getting the minority group
            n_groups = np.max(g_valtrain) + 1
            g_idx = [np.where(g_valtrain == g)[0] for g in range(n_groups)]
            min_g = np.min([len(g) for g in g_idx])
            for g in g_idx:
                np.random.shuffle(g)
            # balancing the groups if balance_val is set to true
            if balance_val:
                x_valtrain = np.concatenate([x_valtrain[g[:min_g]] for g in g_idx])
                y_valtrain = np.concatenate([y_valtrain[g[:min_g]] for g in g_idx])
                g_valtrain = np.concatenate([g_valtrain[g[:min_g]] for g in g_idx])

            x_val = x_val[idx[:n_val]]
            y_val = y_val[idx[:n_val]]
            g_val = g_val[idx[:n_val]]
            #adding training data if add_train is true 
            #note we are only taking training data equal to size of valdation data and concatinating them along with labels and group
            n_train = len(x_valtrain) if add_train else 0

            x_train = np.concatenate([all_embeddings["train"][:n_train], x_valtrain])
            y_train = np.concatenate([all_y["train"][:n_train], y_valtrain])
            g_train = np.concatenate([all_g["train"][:n_train], g_valtrain])
            logger.write(f"bin counts for g_train: {np.bincount(g_train)}\n")
            if preprocess:
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_val = scaler.transform(x_val)
            

            if balance_val and not add_train:
                cls_w_options = [{0: 1., 1: 1.}]  # fix class weights as 1 for both classes
            else:
                cls_w_options = CLASS_WEIGHT_OPTIONS
            # doing logistic regression for different value of c
            # we calculate accuracy on different groups,save the worst one over current hyperparameter(we save sum of worst group accuracy
            # if doing for more then one iteration
            for c in C_OPTIONS:
                for class_weight in cls_w_options:
                    logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear",
                                                class_weight=class_weight)
                    logreg.fit(x_train, y_train)
                    preds_val = logreg.predict(x_val)
                    group_accs = np.array(
                        [(preds_val == y_val)[g_val == g].mean()
                         for g in range(n_groups)])
                    worst_acc = np.min(group_accs)
                    if i == 0:
                        worst_accs[c, class_weight[0], class_weight[1]] = worst_acc
                    else:
                        worst_accs[c, class_weight[0], class_weight[1]] += worst_acc
                    # print(c, class_weight[0], class_weight[1], worst_acc, worst_accs[c, class_weight[0], class_weight[1]])
        ks, vs = list(worst_accs.keys()), list(worst_accs.values())
        # the best hyperparameter that gave best worst group accuarcy
        best_hypers = ks[np.argmax(vs)]
        return best_hypers


    def dfr_on_validation_eval(
            c, w1, w2, all_embeddings, all_y, all_g, num_retrains=20,
            preprocess=True, balance_val=False, add_train=True):
            #Ajayesh adds- 
            #this function is doing last layer retraining on valdation data and evaluating it on test data and returns accuracy on test and train data and mean accuracy on test data

            #input:- c,w1,w2 hyperparameters we got from dfr_on_validation_tune
            #        all_embeddings:embeddings extracted from feature extracter of resnet50 bascically input to last layer
            #        all_y:class labels for ex 0 for landbird and 1 for waterbird
            #        all_g: group label ex landbird on land or waterbird on land and so on(calculated in wb_data.py)
            #        preprocess: flag for weather to standardize embedding to have zero mean and unit standard deviation
            #        balance_val: flag for weather to balance validation data ie to take all of the minority group and take the same size
            #        from other groups and combining them.
            #        add_train: flag for weather to use training data as well (we also tune for class weights in this case)
            #        num_retrain: how many time we retrain last layer
            #output:-accuracy and mean accuracy
        coefs, intercepts = [], [] # to save coefficents and bias terms during iterations
        #preprocessing training data
        if preprocess:
            scaler = StandardScaler()
            scaler.fit(all_embeddings["train"])

        for i in range(num_retrains):
            x_val = all_embeddings["val"]
            y_val = all_y["val"]
            g_val = all_g["val"]
            n_groups = np.max(g_val) + 1
            g_idx = [np.where(g_val == g)[0] for g in range(n_groups)]
            min_g = np.min([len(g) for g in g_idx])
            for g in g_idx:
                np.random.shuffle(g)
            if balance_val:
                x_val = np.concatenate([x_val[g[:min_g]] for g in g_idx])
                y_val = np.concatenate([y_val[g[:min_g]] for g in g_idx])
                g_val = np.concatenate([g_val[g[:min_g]] for g in g_idx])

            n_train = len(x_val) if add_train else 0
            train_idx = np.arange(len(all_embeddings["train"]))
            np.random.shuffle(train_idx)
            train_idx = train_idx[:n_train]

            x_train = np.concatenate(
                [all_embeddings["train"][train_idx], x_val])
            y_train = np.concatenate([all_y["train"][train_idx], y_val])
            g_train = np.concatenate([all_g["train"][train_idx], g_val])
            logger.write(f"bin counts of g_train {np.bincount(g_train)}\n")
            if preprocess:
                x_train = scaler.transform(x_train)

            logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear",
                                        class_weight={0: w1, 1: w2})
            logreg.fit(x_train, y_train)
            coefs.append(logreg.coef_) #save coefficents to use on logistic regression on test data
            intercepts.append(logreg.intercept_) # save bias terms

        x_test = all_embeddings["test"]
        y_test = all_y["test"]
        g_test = all_g["test"]
        logger.write(f"bin counts of g_test: {np.bincount(g_test)}\n") #no of data points in each group

        if preprocess:
            x_test = scaler.transform(x_test)
        logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear",
                                    class_weight={0: w1, 1: w2})
        n_classes = np.max(y_train) + 1
        # the fit is only needed to set up logreg
        logreg.fit(x_train[:n_classes], np.arange(n_classes))
        logreg.coef_ = np.mean(coefs, axis=0) #using mean of coefficents we got multiple iteration in last layer retraining
        logreg.intercept_ = np.mean(intercepts, axis=0) # same with bias term
        preds_test = logreg.predict(x_test)
        preds_train = logreg.predict(x_train)
        n_groups = np.max(g_train) + 1
        test_accs = [(preds_test == y_test)[g_test == g].mean()
                     for g in range(n_groups)]
        test_mean_acc = (preds_test == y_test).mean()
        train_accs = [(preds_train == y_train)[g_train == g].mean()
                      for g in range(n_groups)]
        return test_accs, test_mean_acc, train_accs

        #these both are doing hyper parameter tuning and evaluation with training data these make add_train flag in previous function redundent
    def dfr_train_subset_tune(
            all_embeddings, all_y, all_g, preprocess=True,
            learn_class_weights=False):
        logger.write("dfr train subset_tune\n")
        x_val = all_embeddings["val"]
        y_val = all_y["val"]
        g_val = all_g["val"]

        x_train = all_embeddings["train"]
        y_train = all_y["train"]
        g_train = all_g["train"]

        if preprocess:
            scaler = StandardScaler()
            scaler.fit(x_train)

        n_groups = np.max(g_train) + 1
        g_idx = [np.where(g_train == g)[0] for g in range(n_groups)]
        for g in g_idx:
            np.random.shuffle(g)
        min_g = np.min([len(g) for g in g_idx])
        x_train = np.concatenate([x_train[g[:min_g]] for g in g_idx])
        y_train = np.concatenate([y_train[g[:min_g]] for g in g_idx])
        g_train = np.concatenate([g_train[g[:min_g]] for g in g_idx])
        logger.write(f"bin counts for g_train: {np.bincount(g_train)}\n")
        if preprocess:
            x_train = scaler.transform(x_train)
            x_val = scaler.transform(x_val)

        worst_accs = {}
        if learn_class_weights:
            cls_w_options = CLASS_WEIGHT_OPTIONS
        else:
            cls_w_options = [{0: 1., 1: 1.}]
        for c in C_OPTIONS:
            for class_weight in cls_w_options:
                logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear",
                                            class_weight=class_weight, max_iter=20)
                logreg.fit(x_train, y_train)
                preds_val = logreg.predict(x_val)
                group_accs = np.array(
                    [(preds_val == y_val)[g_val == g].mean() for g in range(n_groups)])
                worst_acc = np.min(group_accs)
                worst_accs[c, class_weight[0], class_weight[1]] = worst_acc
                logger.write(f"{c=}, {class_weight=}, {worst_acc=}, {group_accs=}\n")

        ks, vs = list(worst_accs.keys()), list(worst_accs.values())
        best_hypers = ks[np.argmax(vs)]
        return best_hypers


    def dfr_train_subset_eval(
            c, w1, w2, all_embeddings, all_y, all_g, num_retrains=10,
            preprocess=True):
        logger.write("dfr train subset eval\n")
        coefs, intercepts = [], []
        x_train = all_embeddings["train"]
        scaler = StandardScaler()
        scaler.fit(x_train)

        for i in range(num_retrains):
            x_train = all_embeddings["train"]
            y_train = all_y["train"]
            g_train = all_g["train"]
            n_groups = np.max(g_train) + 1

            g_idx = [np.where(g_train == g)[0] for g in range(n_groups)]
            min_g = np.min([len(g) for g in g_idx])
            for g in g_idx:
                np.random.shuffle(g)
            x_train = np.concatenate([x_train[g[:min_g]] for g in g_idx])
            y_train = np.concatenate([y_train[g[:min_g]] for g in g_idx])
            g_train = np.concatenate([g_train[g[:min_g]] for g in g_idx])
            logger.write(f"bin counts for g_train: {np.bincount(g_train)}\n")

            if preprocess:
                x_train = scaler.transform(x_train)

            logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear",
                                            class_weight={0: w1, 1: w2})
            logreg.fit(x_train, y_train)

            coefs.append(logreg.coef_)
            intercepts.append(logreg.intercept_)

        x_test = all_embeddings["test"]
        y_test = all_y["test"]
        g_test = all_g["test"]

        if preprocess:
            x_test = scaler.transform(x_test)

        logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear")
        n_classes = np.max(y_train) + 1
        # the fit is only needed to set up logreg
        logreg.fit(x_train[:n_classes], np.arange(n_classes))
        logreg.coef_ = np.mean(coefs, axis=0)
        logreg.intercept_ = np.mean(intercepts, axis=0)

        preds_test = logreg.predict(x_test)
        preds_train = logreg.predict(x_train)
        n_groups = np.max(g_train) + 1
        test_accs = [(preds_test == y_test)[g_test == g].mean()
                     for g in range(n_groups)]
        test_mean_acc = (preds_test == y_test).mean()
        train_accs = [(preds_train == y_train)[g_train == g].mean()
                      for g in range(n_groups)]
        return test_accs, test_mean_acc, train_accs



    ## Load data
    target_resolution = (224, 224)
    train_transform = get_transform_cub(target_resolution=target_resolution,
                                        train=True, augment_data=False)
    test_transform = get_transform_cub(target_resolution=target_resolution,
                                       train=False, augment_data=False)
    logger.write("# load trainset\n")
    trainset = WaterBirdsDataset(
        basedir=args.data_dir, split="train", transform=train_transform)
    logger.write("# load testset\n")
    testset = WaterBirdsDataset(
        basedir=args.data_dir, split="test", transform=test_transform)
    logger.write("# load valset\n")
    valset = WaterBirdsDataset(
        basedir=args.data_dir, split="val", transform=test_transform)

    loader_kwargs = {'batch_size': args.batch_size,
                     'num_workers': 4, 'pin_memory': True,
                     "reweight_places": None}
    train_loader = get_loader(
        trainset, train=True, reweight_groups=False, reweight_classes=False,
        **loader_kwargs)
    test_loader = get_loader(
        testset, train=False, reweight_groups=None, reweight_classes=None,
        **loader_kwargs)
    val_loader = get_loader(
        valset, train=False, reweight_groups=None, reweight_classes=None,
        **loader_kwargs)

    # Load model
    logger.write("# Load model\n")
    n_classes = trainset.n_classes
    model = torchvision.models.resnet50(pretrained=False)  # define model without pretraining
    d = model.fc.in_features
    model.fc = torch.nn.Linear(d, n_classes)  # replace fc layer with one suitable for our number of classes
    model.load_state_dict(torch.load(  # bring in the checkpoint image
        args.ckpt_path
    ))
    model.cuda()
    model.eval()

    # Evaluate model
    #evaluation on base model(ERM)
    logger.write("\nBase Model\n")
    base_model_results = {}
    get_yp_func = partial(get_y_p, n_places=trainset.n_places)
    base_model_results["test"] = evaluate(model, test_loader, get_yp_func)
    base_model_results["val"] = evaluate(model, val_loader, get_yp_func)
    base_model_results["train"] = evaluate(model, train_loader, get_yp_func)
    logger.write(f"\n{base_model_results=}\n")

    model.eval()

    # Extract embeddings
    def get_embed(m, x):
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)

        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)

        x = m.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    #processing embedding 
    all_embeddings = {}
    all_y, all_p, all_g = {}, {}, {}
    for name, loader in [("train", train_loader), ("test", test_loader), ("val", val_loader)]:
        all_embeddings[name] = []
        all_y[name], all_p[name], all_g[name] = [], [], []
        for x, y, g, p in tqdm.tqdm(loader):
            with torch.no_grad():
                all_embeddings[name].append(get_embed(model, x.cuda()).detach().cpu().numpy())  # for all 100 x inputs in the batch, capture the 2048 neuron activations of the flattened layer before the fc layer
                all_y[name].append(y.detach().cpu().numpy())
                all_g[name].append(g.detach().cpu().numpy())
                all_p[name].append(p.detach().cpu().numpy())
        all_embeddings[name] = np.vstack(all_embeddings[name])
        all_y[name] = np.concatenate(all_y[name])
        all_g[name] = np.concatenate(all_g[name])
        all_p[name] = np.concatenate(all_p[name])


    # DFR on validation
    # relates to DFR^Val_Tr
    logger.write("DFR on validation TUNE\n")
    dfr_val_results = {}
    c, w1, w2 = dfr_on_validation_tune(
        all_embeddings, all_y, all_g,
        balance_val=args.balance_dfr_val, add_train=not args.notrain_dfr_val)  # NB args.notrain_dfr_val defaults True, then is inverted here so that add_train defaults False
    dfr_val_results["best_hypers"] = (c, w1, w2)
    logger.write(f"Hypers: ({c=}, {w1=}, {w2=})\n")
    logger.write("DFR on validation EVAL\n")
    test_accs, test_mean_acc, train_accs = dfr_on_validation_eval(
            c, w1, w2, all_embeddings, all_y, all_g,
        balance_val=args.balance_dfr_val, add_train=not args.notrain_dfr_val)
    dfr_val_results["test_accs"] = test_accs
    dfr_val_results["train_accs"] = train_accs
    dfr_val_results["test_worst_acc"] = np.min(test_accs)
    dfr_val_results["test_mean_acc"] = test_mean_acc
    logger.write(f"\n{dfr_val_results=}\n")

    # DFR on train subsampled
    # relates to DFR^Tr_Tr
    logger.write("DFR on train subsampled\n")
    dfr_train_results = {}
    c, w1, w2 = dfr_train_subset_tune(
        all_embeddings, all_y, all_g,
        learn_class_weights=args.tune_class_weights_dfr_train)
    dfr_train_results["best_hypers"] = (c, w1, w2)
    logger.write(f"Hypers: ({c=}, {w1=}, {w2=})\n")
    test_accs, test_mean_acc, train_accs = dfr_train_subset_eval(
            c, w1, w2, all_embeddings, all_y, all_g)
    dfr_train_results["test_accs"] = test_accs
    dfr_train_results["train_accs"] = train_accs
    dfr_train_results["test_worst_acc"] = np.min(test_accs)
    dfr_train_results["test_mean_acc"] = test_mean_acc
    logger.write(f"\n{dfr_train_results=}\n")



    all_results = {}
    all_results["base_model_results"] = base_model_results
    all_results["dfr_val_results"] = dfr_val_results
    all_results["dfr_train_results"] = dfr_train_results
    logger.write(f"{all_results=}\n")

    with open(os.path.join(args.output_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(all_results, f)
