from torch import nn

def run(model, weight_init: str, bias_init: str):
    for m in model.modules():
        if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d,nn.Linear)):
            getattr(nn.init, weight_init + '_')(m.weight)
            getattr(nn.init, bias_init + '_')(m.bias)
