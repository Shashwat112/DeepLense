from engine import model_used, epochs, metrics, train_param, train_batch_size, backup_chkpts
from engine.utils import mse, ssim, psnr
from data.loader import train_loader
import torch
import importlib
from tqdm import tqdm
import numpy as np
from typing import List
import os

exp_ID = torch.randint(0,1000,(1,)).item()

train_step = importlib.import_module(f'models.{model_used}.train_step')
loader = train_loader(train_batch_size)
count = os.listdir(f'checkpoints/{model_used}')

def checkpoint(epoch, model, met: list):
    torch.save({'model':model.state_dict(),
                'metrics':met}, f'checkpoints/{model_used}/checkpoint_{exp_ID}_{epoch}.pt')
    count.append(f'checkpoint_{exp_ID}_{epoch}.pt')
    while len(count) > backup_chkpts:
        os.remove(f'checkpoints/{model_used}/{count[0]}')
        count.pop(0)

def model1(chkpt: str = None) -> List[dict]:

    if chkpt is not None:
        train_step.model.load_state_dict(torch.load(f'checkpoints/{model_used}/{chkpt}')['model'])
    met = list()
    for epoch in range(epochs):
        m1,m2,m3 = 0,0,0
        for i,(in_data,lab_data) in enumerate(tqdm(loader, desc=f'Epochs: {epoch+1}/{epochs}')):
            try:
                y_pred = train_step.run(in_data, lab_data)
                m1 += mse(lab_data, y_pred)
                m2 += ssim(lab_data, y_pred)
                m3 += psnr(lab_data, y_pred)
            except KeyboardInterrupt:
                checkpoint(epoch+1, train_step.model, None)
                print('KeyboardInterrupt: Latest Checkpoint saved')
                return met
        met.append({'mse':m1/(i+1), 'ssim':m2/(i+1), 'psnr':m3/(i+1), 'epochs':epoch+1})
        print(f"\nMSE: {met[-1]['mse']} \t  SSIM: {met[-1]['ssim']} \t  PSNR: {met[-1]['psnr']}\n")
        checkpoint(epoch+1, train_step.model, met[-1])
    return met

def model2(chkpt: str = None) -> List[dict]:

    if chkpt is not None:
        train_step.model.load_state_dict(torch.load(f'checkpoints/{model_used}/{chkpt}')['model'])
    met = list()
    for epoch in range(epochs):
        m1,m2,m3 = 0,0,0
        for i,(in_data,lab_data) in enumerate(tqdm(loader, desc=f'Epochs: {epoch+1}/{epochs}')):
            y_pred = train_step.pre_train(in_data, lab_data)
            m1 += mse(lab_data, y_pred)
            m2 += ssim(lab_data, y_pred)
            m3 += psnr(lab_data, y_pred)
        met.append({'mse':m1/(i+1), 'ssim':m2/(i+1), 'psnr':m3/(i+1), 'epochs':epoch+1})
        print(f"\nMSE: {met[-1]['mse']} \t  SSIM: {met[-1]['ssim']} \t  PSNR: {met[-1]['psnr']}\n")
        checkpoint(epoch+1, train_step.model, met[-1])
    return met

def model3(chkpt: str = None) -> List[dict]:

    if chkpt is not None:
        train_step.model.load_state_dict(torch.load(f'checkpoints/{model_used}/{chkpt}')['model'])
    met = list()
    for epoch in range(epochs):
        m1,m2,m3,m4,m5,m6,m7 = 0,0,0,0,0,0,0
        for i,(in_data,lab_data) in enumerate(tqdm(loader, desc=f'Epochs: {epoch+1}/{epochs}')):
            g_pred, d_loss, g_loss, p_real, p_fake = train_step.run(in_data, lab_data)
            m1 += mse(lab_data, g_pred)
            m2 += ssim(lab_data, g_pred)
            m3 += psnr(lab_data, g_pred)
            m4 += d_loss
            m5 += g_loss
            m6 += p_real
            m7 += p_fake
        met.append({'mse':m1/(i+1), 'ssim':m2/(i+1), 'psnr':m3/(i+1), 'd_loss':m4/(i+1), 'g_loss':m5/(i+1), 'p_real':m6/(i+1), 'p_fake':m7/(i+1), 'epochs':epoch+1})
        print(f"\nDLoss: {met[-1]['d_loss']:.10f}\tGLoss: {met[-1]['g_loss']:.10f}\tRealProb: {met[-1]['p_real']:.10f}\tFakeProb: {met[-1]['p_fake']:.10f}\n")
        print(f"\nMSE:\t{met[-1]['mse']}\nSSIM:\t{met[-1]['ssim']}\nPSNR:\t{met[-1]['psnr']}\n")
        checkpoint(epoch+1, train_step.model, met[-1])
    return met

def model4(chkpt: str = None) -> List[dict]:

    if chkpt is not None:
        train_step.model.load_state_dict(torch.load(f'checkpoints/{model_used}/{chkpt}')['model'])
    met = list()
    for epoch in range(epochs):
        L=0
        for i,in_data in enumerate(tqdm(loader, desc=f'Epochs: {epoch+1}/{epochs}')):
            try:
                loss = train_step.run(in_data)
                L += loss
            except KeyboardInterrupt:
                checkpoint(epoch+1, train_step.model, None)
                print('KeyboardInterrupt: Latest Checkpoint saved')
                return met
        met.append({'mse':L/(i+1), 'epochs':epoch+1})
        print(f"\nMSE: {met[-1]['mse']}\n")
        checkpoint(epoch+1, train_step.model, met[-1])
    return met

def run(chkpt: str = None) -> np.ndarray:
    train_met = []
    if model_used == 'srcnn':
        met = model1(chkpt)
    if model_used == 'srgan':
        if train_param.pretrain:
            print('Pretraining of Generator ResNet starting:')
            _met = model2(chkpt)
            train_met.append(_met)
        else:
            _met = None
        met = model3(chkpt)
    if model_used == 'diff_unet':
        met = model4(chkpt)
    for i in metrics+['epochs']:
        train_met.append([dic[i] for dic in met])
    return np.array(train_met).T


