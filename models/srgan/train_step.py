import torch
from models import device
from models.srgan.architecture import Model
from models.srgan import train_param as tp
from engine import optimizer, weight_initializer
from engine.loss import bce, vgg, mse

dev = torch.device('cuda') if torch.cuda.is_available() and device=='gpu' else torch.device('cpu')

G = Model.Generator().to(dev)
D = Model.Discriminator().to(dev)
weight_initializer.run(G, tp.gen_weight_init, tp.gen_bias_init)
weight_initializer.run(D, tp.dis_weight_init, tp.dis_bias_init)
opt_gen = optimizer.run(G.parameters(), tp.gen_optimizer, tp.gen_learning_rate, tp.gen_betas, tp.gen_weight_decay)
opt_dis = optimizer.run(D.parameters(), tp.dis_optimizer, tp.dis_learning_rate, tp.dis_betas, tp.dis_weight_decay)

def pre_train(in_data: torch.Tensor, lab_data: torch.Tensor):
    y_pred = G(in_data.to(dev))
    loss = mse(y_pred,lab_data.to(dev))
    loss.backward()
    with torch.no_grad():
        opt_gen.zero_grad()
        opt_gen.step()
    return y_pred.detach()

def run(in_data: torch.Tensor, lab_data: torch.Tensor):

    d_real = D(lab_data.to(dev))
    g_out = G(in_data.to(dev))
    d_fake = D(g_out.detach())
    dis_loss = (bce(d_real,torch.ones_like(d_real).to(dev)) + bce(d_fake,torch.zeros_like(d_fake).to(dev)))/2
    dis_loss.backward()
    with torch.no_grad():
        opt_dis.zero_grad()
        opt_dis.step()
    
    d_fake = D(g_out)
    adv_loss = bce(d_fake,torch.ones_like(d_fake).to(dev))
    cont_loss = vgg(g_out, lab_data)
    gen_loss = cont_loss + 1e-3*adv_loss
    gen_loss.backward()
    with torch.no_grad():
        opt_gen.zero_grad()
        opt_gen.step()

    return g_out.detach(), dis_loss.item(), gen_loss.item(), torch.mean(d_real).item(), torch.mean(d_fake).item()

