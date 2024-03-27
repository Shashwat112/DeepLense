from torch.utils.data import DataLoader
from data import ddp, exp_name, train_batch_size
if exp_name == 'super_resolution':
    from data import test_batch_size
from data.pipeline import sr_pipe, diff_pipe

if ddp:
    from torch.utils.data.distributed import DistributedSampler
    sampler = DistributedSampler
else:
    sampler = None

if exp_name == 'super_resolution':

    train_loader = lambda bs=train_batch_size: DataLoader(sr_pipe('train'), batch_size=bs, shuffle=True, sampler=sampler)

    test_loader = lambda bs=test_batch_size: DataLoader(sr_pipe('test'), batch_size=bs, shuffle=True)

if exp_name == 'diffusion':

    train_loader = lambda bs=train_batch_size: DataLoader(diff_pipe(), batch_size=bs, shuffle=True, sampler=sampler)
