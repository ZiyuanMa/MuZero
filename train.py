from network import *
import config
import re
import fnmatch
import torch
import pickle
from torch.utils.data import Subset
from tqdm import tqdm
import random
import os
os.environ["OMP_NUM_THREADS"] = "12"

def load_training_set():
    with open('./data_set.pth','rb') as f:
        data_set = pickle.load(f)
        
    return data_set

def load_model():
    network = Network()
    optimizer = optim.SGD(network.parameters(), lr=0.01, weight_decay=config.lr_decay_rate, momentum=config.momentum)
    model_name = None

    for filename in os.listdir('.'):
        if fnmatch.fnmatch(filename, 'model*.pth'):
            model_name = filename

    if not model_name:
        raise RuntimeError('model does not exist')
    
    checkpoint = torch.load(model_name)
    network.load_state_dict(checkpoint['network'])
    network.train()
    optimizer.load_state_dict(checkpoint['optimizer'])
        
    return network, optimizer


def train_network():
    network, optimizer = load_model()
    data_set = load_training_set()

    for i in tqdm(range(config.checkpoint_interval)):

        # sample game data
        sub_data_set = Subset(random.sample(range(len(data_set)), config.batch_size), data_set)
        data_loader = DataLoader(dataset=sub_data_set,
                            num_workers=4,
                            batch_size=config.mini_batch_size,
                            shuffle=True)


        update_weights(optimizer, network, data_loader)

    torch.save({'network': network.state_dict(),
                'optimizer': optimizer.state_dict()
                }, './model'+str(network.steps)+'.pth')

def update_weights(optimizer: torch.optim, network: Network, data_loader):
    network.train()
    optimizer.zero_grad()
    p_loss, v_loss = 0, 0

    for image, actions, target_values, target_rewards, target_policies in data_loader:
        # Initial step, from the real observation.
        net_output = network.initial_inference(image)
        # value, reward, policy_logits, hidden_state = network.initial_inference(image)
        predictions = [(1.0, net_output.value, net_output.reward, net_output.policy_logits)]
        hidden_state = net_output.hidden_state
    # Recurrent steps, from action and previous hidden state.
        for action in actions:
            net_output = network.recurrent_inference(hidden_state, action)
            predictions.append((1.0 / len(actions), net_output.value, net_output.reward, net_output.policy_logits))
            hidden_state = net_output.hidden_state

        for prediction, target_value, target_reward, target_policy in zip(predictions, target_values, target_rewards, target_policies):
            _ , value, reward, policy_logits = prediction


            p_loss += torch.mean(torch.sum(-target_policy * torch.log(policy_logits), dim=1))
            v_loss += torch.mean(torch.sum((target_value - value) ** 2, dim=1))
  
  
    loss = (p_loss + v_loss)
    loss.backward()
    optimizer.step()
    network.steps += 1
    print('p_loss %f v_loss %f' % (p_loss, v_loss))

if __name__ == '__main__':

    train_network()