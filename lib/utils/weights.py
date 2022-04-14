import torch


def KaiMingInit(net):
    """Kaiming Init layer parameters."""
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, a=0.2)  # slope = 0.2 in the original implementation
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1)
        elif isinstance(m, torch.nn.Linear):
            # torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.normal_(m.weight, std=1e-3)  # 1e-2 for global, 1e-3 default
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)


def load_pretrained_backbone(prefix, model, pth_path):
    # load pretrained checkpoint
    from collections import OrderedDict
    checkpoint = torch.load(pth_path, map_location=lambda storage, loc: storage.cuda())
    state_dict = checkpoint['model']

    pretrained_dict = OrderedDict()
    for k, v in state_dict.items():
        pretrained_dict[prefix+k] = v

    # current model
    model_dict = model.state_dict()

    # compare keys and update value
    pretrained_dict_can_load = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    pretrained_dict_cannot_load = [k for k, v in model_dict.items() if k not in pretrained_dict]
    print("pretrained_dict_cannot_load", pretrained_dict_cannot_load)
    print("Pretrained: {}/ Loaded: {}/ Cannot loaded: {} VS Current model: {}".format(len(pretrained_dict),
                                                                                      len(pretrained_dict_can_load),
                                                                                      len(pretrained_dict_cannot_load),
                                                                                      len(model_dict)))
    model_dict.update(pretrained_dict_can_load)
    model.load_state_dict(model_dict)
    print("Load pretrained backbone done!")


def load_checkpoint(model, pth_path):
    # load checkpoint
    checkpoint = torch.load(pth_path, map_location=lambda storage, loc: storage.cuda())
    state_dict = checkpoint['model']
    model_dict = model.state_dict()
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    print("Load checkpoint done!")


def save_checkpoint(state, file_path):
    print("Saving to {}".format(file_path))
    torch.save(state, file_path)
