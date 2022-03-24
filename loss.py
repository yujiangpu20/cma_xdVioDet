import torch


def CLAS(logits, label, seq_len, criterion, is_topk=True):
    logits = logits.squeeze()
    instance_logits = torch.zeros(0).cuda()
    for i in range(logits.shape[0]):
        if is_topk:
            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(seq_len[i]//16+1), largest=True)
            tmp = torch.mean(tmp).view(1)
        else:
            tmp = torch.mean(logits[i, :seq_len[i]]).view(1)
        instance_logits = torch.cat((instance_logits, tmp))

    instance_logits = torch.sigmoid(instance_logits)
    clsloss = criterion(instance_logits, label)
    return clsloss


def temporal_smooth(arr):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]
    loss = torch.sum((arr2-arr)**2)
    return loss


def temporal_sparsity(arr):
    loss = torch.sum(arr)
    return loss


def Smooth_Sparsity(logits, seq_len, lamda=8e-5):
    smooth_mse = []
    spar_mse = []
    for i in range(logits.shape[0]):
        tmp_logits = logits[i][:seq_len[i]]
        sm_mse = temporal_smooth(tmp_logits)
        sp_mse = torch.sum(tmp_logits)
        smooth_mse.append(sm_mse)
        spar_mse.append(sp_mse)
    smooth_mse = sum(smooth_mse) / len(smooth_mse)
    spar_mse = sum(spar_mse) / len(spar_mse)
    return (smooth_mse+spar_mse) * lamda