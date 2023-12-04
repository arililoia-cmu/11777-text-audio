import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class TripletHead(nn.Module):
    def __init__(self, margin):
        super(TripletHead, self).__init__()
        self.margin = margin
        self.relu = nn.ReLU()

    def forward(self, anchor, positive, negative, size_average=True):
        cosine_positive = nn.CosineSimilarity(dim=-1)(anchor, positive)
        cosine_negative = nn.CosineSimilarity(dim=-1)(anchor, negative)
        losses = self.relu(self.margin - cosine_positive + cosine_negative)
        return losses.mean()

class CLIPHead(nn.Module):
    def __init__(self, logit_scale, combine=0.0):
        super(CLIPHead, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.logit_scale = logit_scale
        self.combine = combine

    def forward(self, h1, h2, cluster_mask=None):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            h1 = SyncFunction.apply(h1)
            h2 = SyncFunction.apply(h2)
        h1 = F.normalize(h1, dim=-1)
        h2 = F.normalize(h2, dim=-1)
        if h1.ndim == 2:  # one cluster (shape: batch_size x dim)
            loss, correct = self.compute_loss(h1, h2)
        else:  # multiple clusters (shape: num_cluster x batch_size x dim)
            if self.combine > 0:
                combine_h1 = h1.permute(1, 0, 2) * cluster_mask.unsqueeze(-1)
                combine_h1 = combine_h1.reshape(combine_h1.shape[0], -1)
                combine_h2 = h2.permute(1, 0, 2) * cluster_mask.unsqueeze(-1)
                combine_h2 = combine_h2.reshape(combine_h2.shape[0], -1)
                combine_loss, combine_correct = self.compute_loss(combine_h1, combine_h2)
            if self.combine == 1:
                loss = combine_loss
                correct = combine_correct
            else:
                loss, correct = self.compute_loss(h1, h2, cluster_mask)
                loss = self.combine * combine_loss + (1 - self.combine) * loss
                correct.append(combine_correct)
        return loss, correct
    
    def compute_loss(self, h1, h2, cluster_mask=None):
        device = h1.device
        temperature = torch.clamp(self.logit_scale.exp(), max=100)
        if cluster_mask is not None:
            logits = torch.einsum('knc,kmc->knm', [h1, h2]) * temperature.to(device)
            M, N = logits.shape[0:2]  # cluster size, batch size per GPU
            logits = logits.permute(1, 2, 0)
            labels = torch.arange(N, dtype=torch.long, device=device).unsqueeze(-1).repeat(1, M)
            loss = F.cross_entropy(logits, labels, reduction='none')
            loss = loss * cluster_mask
            loss = loss.sum() / cluster_mask.sum()
            correct = self.correct(logits, labels, cluster_mask)
        else:
            logits = torch.einsum('nc,mc->nm', [h1, h2]) * temperature.to(device)
            N = logits.shape[0]  # batch size per GPU
            labels = torch.arange(N, dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, labels)
            correct = self.correct(logits, labels)
        return loss, correct
    
    def correct(self, logits, target, cluster_mask=None):
        y_pred = logits.max(dim=1)[1]
        if cluster_mask is not None:
            correct = torch.sum((y_pred == target) * cluster_mask, dim=0)
            correct = correct.tolist()
        else:
            correct = torch.sum(y_pred == target)
        return correct

class ContrastiveHead(nn.Module):
    def __init__(self, temperature=0.2):
        super(ContrastiveHead, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, pos, neg):
        """
        pos: batch x 1 (sim score)
        neg: batch x neg (sim score)
        """
        N = pos.size(0)
        logits = torch.cat((pos, neg), dim=1)
        logits /= self.temperature
        labels = torch.zeros((N, ), dtype=torch.long).cuda()
        loss = self.criterion(logits, labels)
        return loss

class ClsHead(nn.Module):
    """Simplest classifier head, with only one fc layer.
    """
    def __init__(self, in_channels, num_classes=1054, with_avg_pool=False):
        super(ClsHead, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fc_cls = nn.Linear(in_channels, num_classes, bias=False) # class centroid
        self.sigmoid = nn.Sigmoid()
        self.loss_fn = nn.BCELoss()

    def forward(self, x,y):
        if self.with_avg_pool:
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
        output = self.fc_cls(x)
        logits = self.sigmoid(output)
        loss = self.loss_fn(logits,y)
        return loss

class SyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]
        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)
        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input = grad_input.contiguous() # contiguous error
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)
        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]
