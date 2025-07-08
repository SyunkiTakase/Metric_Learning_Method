

config = {
    "epoch": 20,
    "batch_size": 128,
    "lr": 1e-3,
    "img_size": 32,
    "dataset": "cifar100",
    "margin": 0.2,
    "scale": None,
    "method": "TripletLoss",
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
#     "margin": 0.2,
#     "scale": None,
#     "method": "TripletLoss",
#     "arch": "ResNet18",
#     "_lambda": 0.2, # hard_tripletsをTrueにする場合は0.2くらいにする
#     "num_dim": None,
#     "hard_triplets": True,
#     "easy_margin": None,
# }