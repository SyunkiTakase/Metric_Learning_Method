

config = {
    "epoch": 20,
    "batch_size": 128,
    "lr": 1e-3,
    "img_size": 32,
    "dataset": "cifar10",
    "margin": 1.0,
    "scale": None,
    "method": "SiameseNetwork",
    "arch": "ResNet18",
    "_lambda": 1.0,
    "num_dim": None,
    "hard_triplets": None,
    "easy_margin": None,
}


# config = {
#     "epoch": 20,
#     "batch_size": 128,
#     "lr": 1e-3,
#     "img_size": 32,
#     "dataset": "cifar10",
#     "margin": 1.0,
#     "scale": None,
#     "method": "SiameseNetwork",
#     "arch": "Xception",
#     "_lambda": 1.0,
#     "num_dim": None,
#     "hard_triplets": None,
#     "easy_margin": None,
# }
