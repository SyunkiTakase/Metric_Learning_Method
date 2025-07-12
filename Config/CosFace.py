

config = {
    "epoch": 20,
    "batch_size": 128,
    "lr": 1e-1,
    "weight_decay": 5e-4,
    "momentum": 0.9,
    "img_size": 32,
    "dataset": "cifar100",
    "margin": 0.35,
    "scale": 30,
    "method": "CosFace",
    "arch": "ResNet18",
    "_lambda": None,
    "num_dim": 512,
    "hard_triplets": None,
    "easy_margin": None,
}
