# import torch
# import timm
# from torch import nn
#
# class PrintLayer(nn.Module):
#    def __init__(self, layer):
#        super(PrintLayer, self).__init__()
#        self.layer = layer
#        self.layer.register_forward_hook(self.print_sizes)
#
#    def print_sizes(self, module, input, output):
#        print(module.__class__.__name__, 'input size: ', input[0].size())
#        print(module.__class__.__name__, 'output size: ', output.size())
#
#    def forward(self, x):
#        return self.layer(x)
#
#    def __getattr__(self, name):
#        try:
#            return super().__getattr__(name)
#        except AttributeError:
#            return getattr(self.layer, name)
#
# model_name = 'twins_pcpvt_small'
# model = timm.create_model(model_name, pretrained=True)
#
# model.blocks = model.blocks[:3] # Only use the first 3 blocks
# model.norm = torch.nn.Identity() # Remove the normalization layer
# model.head = torch.nn.Identity() # Remove the head layer
# model.head_drop = torch.nn.Identity() # Remove the head layer
# model.patch_embeds[3] = torch.nn.Identity() # Remove the head layer
# model.pos_block[3] = torch.nn.Identity() # Remove the head layer
#
# print(model)
#
# def replace_with_printlayer(module):
#    for name, layer in module.named_children():
#        if list(layer.children()):
#            replace_with_printlayer(layer)
#        else:
#            setattr(module, name, PrintLayer(layer))
#
# replace_with_printlayer(model)
#
# input = torch.randn(1, 3, 400, 400)
# output = model(input)

import torch
import torch.nn as nn
import timm


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x, y):
        print(x.shape)
        d = x.clone()
        d = d.reshape(x.shape[0], x.shape[2], y[0], y[1])
        print(d.shape)
        return x


def print_output_size(module, input, output):
    print(module.__class__.__name__, "output size: ", output.size())


model_name = "twins_pcpvt_small"
model = timm.create_model(model_name, pretrained=True)

model.blocks = model.blocks[:3]  # Only use the first 3 blocks
model.norm = torch.nn.Identity()  # Remove the normalization layer
model.head = torch.nn.Identity()  # Remove the head layer
model.head_drop = torch.nn.Identity()
model.patch_embeds[3] = torch.nn.Identity()
model.pos_block[3] = torch.nn.Identity()
model.blocks[0].add_module("print", PrintLayer())
model.blocks[1].add_module("print", PrintLayer())
model.blocks[2].add_module("print", PrintLayer())
print(model)

input = torch.randn(1, 3, 400, 400)
output = model(input)
