import torch


class OptimizerManager(object):
    def __init__(self, optimizer, lr, lr_decay, milestones, model, **kwargs):
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=kwargs['momentum'])
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        if milestones is not None:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[int(m) for m in milestones], gamma=lr_decay)
        else:
            class NullScheduler(object):
                def step(self): pass
            scheduler = NullScheduler()

        self.optimizer = optimizer
        self.scheduler = scheduler

    def step(self):
        self.optimizer.step()
        self.scheduler.step()

    def zero_grad(self):
        return self.optimizer.zero_grad()

    @property
    def lr(self):
        return self.optimizer.param_groups[0]['lr']
