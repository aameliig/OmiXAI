# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

import torch
import captum
from captum.attr import IntegratedGradients, InputXGradient, GuidedBackprop, Deconvolution
from torch_geometric.explain import Explainer, GNNExplainer, CaptumExplainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cnn_interpretation_pipeline(model, loader_test, loader_train, width, save_filename, algorithm='IntegratedGradients', need_return=1):
    # set the algorithm and config
    if algorithm == 'InputXGradient':
      explain = InputXGradient(model)

    elif algorithm == 'Deconvolution':
      explain = Deconvolution(model)

    elif algorithm == 'GuidedBackprop':
      explain = GuidedBackprop(model)

    else:
      explain = IntegratedGradients(model)


    mean1 = np.zeros(1950, dtype=float)
    cnt = 0

    # for train data
    for x, y_true in tqdm(loader_train):
        # make prediction
        x, y_true = x.to(device), y_true.to(device).long()
        output = model(x)
        pred = torch.argmax(output, dim=1).reshape(1, width)

        # find True Positive indices
        idxs = []
        for i in range(width):
            if pred[0][i] == y_true[0][i] and y_true[0][i] == 1:
                idxs.append(i)

        # interpretation
        if algorithm =='IntegratedGradients':
            explanation = explain.attribute(x, target=1, n_steps=1)
        else:
            explanation = explain.attribute(x, target=1)
        explanation = torch.squeeze(explanation, dim=0)

        if explanation[idxs, :].shape != (0, 1950):
            explanation = torch.mean(explanation[idxs, :], dim=0)
            explanation = explanation.cpu().detach().numpy()
            mean1 += explanation
            cnt += 1


    # for test data
    for x, y_true in tqdm(loader_test):
        # make prediction
        x, y_true = x.to(device), y_true.to(device).long()
        output = model(x)
        pred = torch.argmax(output, dim=1).reshape(1, width)

        # find True Positive indices
        idxs = []
        for i in range(width):
            if pred[0][i] == y_true[0][i] and y_true[0][i] == 1:
                idxs.append(i)

        # interpretation
        if algorithm =='IntegratedGradients':
            explanation = explain.attribute(x, target=1, n_steps=1)
        else:
            explanation = explain.attribute(x, target=1)
        explanation = torch.squeeze(explanation, dim=0)

        if explanation[idxs, :].shape != (0, 1950):
            explanation = torch.mean(explanation[idxs, :], dim=0)
            explanation = explanation.cpu().detach().numpy()
            mean1 += explanation
            cnt += 1

    print('done interpretation')

    # average
    mean = mean1 / cnt
    mean = torch.from_numpy(mean)
    print(f'Averaged tensor shape: {mean.shape}')
    print(f'Averaged tensor: {mean}')

    torch.save(mean, f'{save_filename}.pt')
    print('Interpretation result is an averaged tensor. It is saved as:')
    print(f'{save_filename}.pt')

    if need_return:
      return mean
    else:
      return

def get_ranked_features(features_weights):
    features = features_weights
    p_deviation = pd.DataFrame()

    for column in features_weights.columns:
        if column == 'Unnamed: 0':
            continue

        mean = features_weights[column].mean()
        p_deviation[f'{column}_p_deviation'] = (((features_weights[column] - mean) / mean) * 100).abs() # считаем процентное среднее

    p_deviation['mean_deviation'] = p_deviation.mean(axis=1)
    features_range = p_deviation[['mean_deviation']].sort_values(by='mean_deviation', ascending=False)

    return features_range



def graph_interpretation_pipeline(model, loader_test, width, algorithm, save_filename, need_return=1):
    # set the algorithm and config
    if algorithm == 'GNNExplainer':
      explain = GNNExplainer(epochs=50)
    else:
      explain = CaptumExplainer(algorithm)


    explainer = Explainer(model=model,
                          algorithm=explain,
                          explanation_type='model',
                          node_mask_type='attributes',
                          edge_mask_type='object',
                          model_config=dict(
                                mode='multiclass_classification',
                                task_level='node',
                                return_type='probs',
                                ),)
    # to store results
    mean_1 = np.zeros(1950, dtype=float)
    cnt= 0

    for dt in tqdm(loader_test):
        # make prediction
        x, edge, y = dt.x.cuda(), dt.edge_index.cuda(), dt.y.cuda().long()
        valid_edges = (edge < width).all(dim=0)
        edge = edge[:, valid_edges]

        output = model(x, edge.squeeze())
        pred = torch.argmax(output, dim=-1)

        # find True Positive indices
        idxs = []
        for i in range(width):
            if pred[0][i] == y[0][i] and y[0][i] == 1:
                idxs.append(i)

        # interpretation
        explanation = explainer(x.squeeze(), edge)
        node_mask = explanation.node_mask

        # select results of True Positiv indicies only
        if node_mask[idxs, :].shape != (0, 1950):
            node_mask = torch.mean(node_mask[idxs, :], dim=0)
            node_mask = node_mask.cpu().detach().numpy()
            mean_1 += node_mask
            cnt += 1


    print('done interpretation')

    # average
    mean = mean_1 / cnt
    mean = torch.from_numpy(mean)
    print(f'Averaged tensor shape: {mean.shape}')
    print(f'Averaged tensor: {mean}')

    torch.save(mean, f'{save_filename}.pt')
    print('Interpretation result is an averaged tensor. It is saved as:')
    print(f'{save_filename}.pt')

    if need_return:
      return mean
    else:
      return