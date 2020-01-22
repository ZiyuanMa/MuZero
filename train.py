from network import *

def load_training_data():
    with open('./data.pth','wb') as f:
        data = pickle.load(f)
    return data
    