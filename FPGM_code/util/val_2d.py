import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0

def calculate_metric_percase_(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum()>0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0

def test_single_slice(image, label, model, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
   
    slice = image
    x, y = slice.shape[0], slice.shape[1]
    slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
    input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
    model.eval()
    with torch.no_grad():
        output = model(input)
        if len(output)>1:
            output = output[0]
        out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        prediction = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase_(prediction == i, label == i))
    return metric_list

def test_single_volume(image, label, model, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        model.eval()
        with torch.no_grad():
            output = model(input)
            if len(output)>1:
                output = output[0]
            out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list

def test_isic_images(image, label, model, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    
    slice = image
    x, y = slice.shape[1], slice.shape[2]
    slice = zoom(slice, (1, patch_size[0] / x, patch_size[1] / y), order=0)
    label = zoom(label, (patch_size[0] / x, patch_size[1] / y), order=0)
    input = torch.from_numpy(slice).unsqueeze(0).float().cuda()
    model.eval()
    with torch.no_grad():
        output = model(input)
        if len(output)>1:
            output = output[0]
        out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        #pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        prediction = out
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list

def test_promise_images(image, label, model, classes, patch_size=[256, 256]):
    image, label = image.cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    
    slice = image
    x, y = slice.shape[1], slice.shape[2]
    slice = zoom(slice, (1, patch_size[0] / x, patch_size[1] / y), order=0)
    label = zoom(label, (patch_size[0] / x, patch_size[1] / y), order=0)
    input = torch.from_numpy(slice).unsqueeze(0).float().cuda()
    model.eval()
    with torch.no_grad():
        output = model(input)
        if len(output)>1:
            output = output[0]
        out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        #pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        prediction = out
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list
