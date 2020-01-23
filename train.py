from network import *
import config
import re
import fnmatch
import torch
from torch.utils.data import Subset
import os
os.environ["OMP_NUM_THREADS"] = "12"

def load_training_set():
    with open('./data.pth','rb') as f:
        data_set = pickle.load(f)
        
    return data_set

def load_network():
    network = Network()
    model_name = None

    for filename in os.listdir('.'):
        if fnmatch.fnmatch(filename, 'model*.pth'):
            model_name = filename

    if not model_name:
        raise RuntimeError('model does not exist')
    
    checkpoint = torch.load(model_name)
    network.load_state_dict(checkpoint)
        
    return network

def load_optimizer():
    with open('./optimizer.pth','rb') as f:
        optimizer = pickle.load(f)
        
    return optimizer

def train_network(storage: SharedStorage, replay_buffer: ReplayBuffer):
    network = load_network()
    optimizer = load_optimizer()
    data_set = load_training_set()

    for i in tqdm(range(config.checkpoint_interval)):
        batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
        sub_data_set = Subset(random.sample(range(len(data_set)), config.batch_size), data_set)
        data_loader = DataLoader(dataset=sub_data_set,
                            num_workers=4,
                            batch_size=config.mini_batch_size,
                            shuffle=True)
        update_weights(optimizer, network, data_loader)

    torch.save(network.state_dict(), './model'+str(network.steps)+'.pth')

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