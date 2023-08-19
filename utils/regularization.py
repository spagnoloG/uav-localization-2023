import timm
import torch

model = timm.create_model("twins_pcpvt_small", pretrained=True)


dropout_dict = {
    "dropout": 0.1,
    "attn_drop": 0.2,
    "proj_drop": 0.3,
    "head_drop": 0.4,
    "mlp_drop1": 0.5,
    "mlp_drop2": 0.6,
    "pos_drops": 0.7,
}


def set_dropout_values(model, dropout_values):
    for module in model.modules():
        if hasattr(module, "attn_drop"):
            print("This should be called")
            module.attn_drop.p = dropout_values.get("attn_drop", module.attn_drop.p)
        if hasattr(module, "proj_drop"):
            module.proj_drop.p = dropout_values.get("proj_drop", module.proj_drop.p)
        if hasattr(module, "head_drop"):
            module.head_drop.p = dropout_values.get("head_drop", module.head_drop.p)
        if hasattr(module, "drop1"):
            module.drop1.p = dropout_values.get("mlp_drop1", module.drop1.p)
        if hasattr(module, "drop2"):
            module.drop2.p = dropout_values.get("mlp_drop2", module.drop2.p)
        if hasattr(module, "pos_drops"):
            for drop in module.pos_drops:
                drop.p = dropout_values.get("pos_drops", drop.p)


set_all_dropout(model, dropout_dict)
print(model)
