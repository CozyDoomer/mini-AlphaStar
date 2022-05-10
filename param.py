# param for some configs, for ease use of changing different servers
# also ease of use for experiments

"""whether is running on server, on server meaning use GPU with larger memoary"""
on_server = False

"""The replay path"""
replay_path = "/home/cozy/Documents/projects/sc2_rl/mini-AlphaStar/data/replays/abyssal-reef-dataset-4500/"
# replay_path = "/home/cozy/Documents/projects/sc2_rl/mini-AlphaStar/data/replays/selected-liuruoze/"
# replay_path = "/home/cozy/Documents/projects/sc2_rl/mini-AlphaStar/data/replays/3.16.1/"
# replay_path = "/home/liuruoze/data4/mini-AlphaStar/data/filtered_replays_1/"
# replay_path = "/home/liuruoze/mini-AlphaStar/data/filtered_replays_1/"

"""The mini scale used in hyperparameter"""
Batch_Scale = 16
Seq_Scale = 16
Select_Scale = 4

handle_cuda_error = False

skip_entity_list = False
skip_autoregressive_embedding = False
