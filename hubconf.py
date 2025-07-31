import torch

dependencies = ['torch', 'torchvision', 'PIL', 'monai']


def medsam_unetr(pretrained=True, **kwargs):
    from src.unetr import medsam_unetr
    model = medsam_unetr()

    if pretrained: 
        state_dict = torch.hub.load_state_dict_from_url(
            'https://github.com/pfrwilson/microUS-seg/releases/download/v0.0.0/medsam_unetr_microsegnet.pth', map_location='cpu'
        )
        print(model.load_state_dict(state_dict))

    return model
    
     
