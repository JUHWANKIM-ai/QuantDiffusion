import torch
import numpy as np
from torch import nn

class Loss(nn.Module):
    def __init__(self, name, reg=1.0, transform=lambda x: x, threshold=10., backward=False, norm_foo=lambda x: x, seed=None):
        super(Loss, self).__init__()
        self.name = name
        self.reg = reg
        self.transform = transform
        self.threshold = threshold
        self.backward = backward
        self.norm_foo = norm_foo
        self.seed = seed

    def forward(self, x_fake):
        self.loss_componentwise = self.compute(x_fake)        
        return {'abs_loss': self.reg * self.loss_componentwise['abs_loss'].mean(), 'rel_loss': self.reg * self.loss_componentwise['rel_loss'].mean()}


    def compute(self, x_fake):
        raise NotImplementedError()
    
    @property
    def success(self):
        return torch.all(self.loss_componentwise <= self.threshold)



class VARLoss(Loss):
    def __init__(self, x_real, alpha=0.05, **kwargs):
        name = kwargs.pop('name')
        super(VARLoss, self).__init__(name=name)
        self.alpha = alpha
        self.var = tail_metric(x=x_real, alpha=self.alpha, statistic='var')

    def compute(self, x_fake):
        abs_loss_list = []
        rel_loss_list = []
        var_fake = tail_metric(x=x_fake, alpha=self.alpha, statistic='var')
        for i in range(x_fake.shape[2]):
            for t in range(x_fake.shape[1]):
                abs_metric = torch.abs(var_fake[i][t] - self.var[i][t].to(x_fake.device))
                eps = 1e-8  # 분모 0 방지용
                rel_metric = abs_metric / (torch.abs(self.var[i][t].to(x_fake.device)) + eps)

                abs_loss_list.append(abs_metric)
                rel_loss_list.append(rel_metric)

        abs_loss_componentwise = torch.stack(abs_loss_list)
        rel_loss_componentwise = torch.stack(rel_loss_list)

        return {
            'abs_loss': abs_loss_componentwise,
            'rel_loss': rel_loss_componentwise
        }

class ESLoss(Loss):
    def __init__(self, x_real, alpha=0.05, **kwargs):
        name = kwargs.pop('name')
        super(ESLoss, self).__init__(name=name)
        self.alpha = alpha
        self.var = tail_metric(x=x_real, alpha=self.alpha, statistic='es')

    def compute(self, x_fake):
        abs_loss_list = []
        rel_loss_list = []
        
        var_fake = tail_metric(x=x_fake, alpha=self.alpha, statistic='es')
        for i in range(x_fake.shape[2]):
            for t in range(x_fake.shape[1]):
                abs_metric = torch.abs(var_fake[i][t] - self.var[i][t].to(x_fake.device))
                eps = 1e-8  # 분모 0 방지용
                rel_metric = abs_metric / (torch.abs(self.var[i][t].to(x_fake.device)) + eps)
                
                abs_loss_list.append(abs_metric)
                rel_loss_list.append(rel_metric)
        abs_loss_componentwise = torch.stack(abs_loss_list)
        rel_loss_componentwise = torch.stack(rel_loss_list)

        return {
            'abs_loss': abs_loss_componentwise,
            'rel_loss': rel_loss_componentwise
        }

def tail_metric(x, alpha, statistic):
    res = list()
    for i in range(x.shape[2]):
        tmp_res = list()
        # Exclude the initial point
        for t in range(x.shape[1]):
            x_ti = x[:, t, i]
            sorted_arr, _ = torch.sort(x_ti)
            var_alpha_index = int(alpha * len(sorted_arr))
            var_alpha = sorted_arr[var_alpha_index]
            if statistic == "es":
                es_values = sorted_arr[:var_alpha_index + 1]
                es_alpha = es_values.mean()
                tmp_res.append(es_alpha)
            else:
                tmp_res.append(var_alpha)
        res.append(tmp_res)
    return res
