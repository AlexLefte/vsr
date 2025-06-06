from .isr_model import ISRModel
from .vsr_model import VSRModel
from .vsrgan_model import VSRGANModel

# register vsr model
vsr_model_lst = [
    'frvsr',
]

# register vsrgan model
vsrgan_model_lst = [
    'tecogan',
]

def define_model(opt):
    if opt['model']['name'].lower() in vsr_model_lst:
        model = VSRModel(opt)
    elif opt['model']['name'].lower() in vsrgan_model_lst:
        model = VSRGANModel(opt)
    elif opt['model']['name'].lower() == 'isr':
        model = ISRModel(opt)
    else:
        raise ValueError('Unrecognized model: {}'.format(opt['model']['name']))

    return model
