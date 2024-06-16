def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
