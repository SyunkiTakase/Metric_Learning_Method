

config = {
    "epoch": 20,
    "batch_size": 128,
    "lr": 1e-3,
    "weight_decay": 5e-4,
    "momentum": 0.9,
    "img_size": 224,
    "dataset": "cifar10",
    "margin": 4,
    "scale": None,
    "method": "SphereFace",
    "arch": "ResNet18",
    "_lambda": None,
    "num_dim": 512,
    "hard_triplets": None,
    "easy_margin": None,
    "vis_featspace": True
}
