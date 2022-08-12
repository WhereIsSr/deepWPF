import numpy as np
import common.metrics as metrics
import logging
import torch


def evaluate(predictions, grounds, raw_data, cfg):
    pre = np.array(predictions)
    gt = np.array(grounds)
    pre = np.sum(pre, axis=0)
    gt = np.sum(gt, axis=0)

    day_len = cfg["dataset"]["day_len"]
    day_acc = []
    for idx in range(0, pre.shape[0]):
        acc = 1 - metrics.rmse(pre[idx, -day_len:, -1], gt[idx, -day_len:, -1]) / (cfg["dataset"]["num_turbines"] * 1000)
        if acc != acc:
            continue
        day_acc.append(acc)
    day_acc = np.array(day_acc).mean()
    logging.info('Day accuracy:  {:.4f}%'.format(day_acc * 100))

    overall_mae, overall_rmse = metrics.regressor_detailed_scores(predictions, grounds, raw_data, cfg)
    logging.info('\n \t RMSE: {}, MAE: {}'.format(overall_rmse, overall_mae))

    total_score = (overall_mae + overall_rmse) / 2
    logging.info(total_score)


def val(model, dataloader, loss, device):
    losses = []
    for idx, (x, y) in enumerate(dataloader):
        x = x.to(device).type(torch.float32)
        y = y.to(device).type(torch.float32)[:, :, 9:10]
        y_pre = model(x)
        losses.append(loss(y_pre, y).item())
    return np.average(losses)


def test(model, dataset, device):
    data, loader = dataset
    predictions = []
    true_list = []
    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x = x.to(device).type(torch.float32)
            y = y.type(torch.float32)[:, :, 9:10]
            y_pre = model(x)
            predictions.append(np.array(y_pre.to(torch.device("cpu"))))
            true_list.append(np.array(y))
    predictions = np.array(predictions)
    predictions = predictions.reshape((-1, predictions[0].shape[-2], predictions[0].shape[-1]))
    true_list = np.array(true_list)
    true_list = true_list.reshape((-1, true_list.shape[-2], true_list.shape[-1]))
    return data.inverse_transform(predictions), data.inverse_transform(true_list), data.get_rawData()

