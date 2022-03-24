from sklearn.metrics import auc, precision_recall_curve
import numpy as np
import torch


def test(dataloader, model, gt):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).cuda()

        for i, inputs in enumerate(dataloader):
            inputs = inputs.cuda()

            logits = model(inputs)
            logits = torch.mean(logits, 0)
            pred = torch.cat((pred, logits))

        pred = list(pred.cpu().detach().numpy())
        precision, recall, th = precision_recall_curve(list(gt), np.repeat(pred, 16))
        pr_auc = auc(recall, precision)

        return pr_auc


