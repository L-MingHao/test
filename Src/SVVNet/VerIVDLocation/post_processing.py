import torch


def post_processing(pred_, target_D, device):
    assert pred_.dim() == 5, 'should be 5D tensor'
    batch_size, _, d, h, w = pred_.shape
    # td, th, tw = target.shape[-3:]
    td = target_D
    th = h
    tw = w

    pred = torch.zeros(1, 1, td, th, tw).to(device)

    pred_a = pred_[0]
    pred_b = pred_[1]
    pred[:, :, :d, :, :] += pred_a
    pred[:, :, td - d:, :, :] += pred_b
    pred[:, :, td - d:d, :, :] /= 2.

    return pred
