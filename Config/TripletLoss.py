

config = {
    "epoch": 20,
    "batch_size": 128,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "momentum": 0.9,
    "img_size": 224,
    "dataset": "cifar10",
    "margin": 1.0,
    "scale": None,
    "method": "TripletLoss",
    "arch": "ResNet18",
    "_lambda": 1.0,
    "num_dim": None,
    "hard_triplets": None,
    "easy_margin": None,
    "vis_featspace": True
}


# config = {
#     "epoch": 20,
#     "batch_size": 128,
#     "lr": 1e-3,
#     "weight_decay": 1e-4,
#     "momentum": 0.9,
#     "img_size": 224,
#     "dataset": "cifar10",
#     "margin": 1.0,
#     "scale": None,
#     "method": "TripletLoss",
#     "arch": "ResNet18",
#     "_lambda": 0.2,
#     "num_dim": None,
#     "hard_triplets": True,
#     "easy_margin": None,
#     "vis_featspace": True
# }