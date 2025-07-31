dependencies = ['torch', 'torchvision', 'PIL', 'monai']


def medsam_unetr(pretrained=True, **kwargs):
    from src.unetr import medsam_unetr
    model = medsam_unetr()

    return model
    
     