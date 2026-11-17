def freeze_parameter(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if any(part in name for part in ("decoder", "joint", "prompt")):
            parameter.requires_grad = True
    for layer in model.encoder.layers[-8:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
     
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M")