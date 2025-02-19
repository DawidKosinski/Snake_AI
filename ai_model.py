import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn



class DeepQNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save_model(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class NeuralNetworkTrainer:
    def __init__(self, model, lr, discount_factor):
        self.learning_rate = lr
        self.discount_factor = discount_factor
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, is_done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            is_done = (is_done, )

        # 1: predicted Q values with current state
        predicted_q_values = self.model(state)

        target_q_values = predicted_q_values.clone()
        for idx in range(len(is_done)):
            Q_new = reward[idx]
            if not is_done[idx]:
                Q_new = reward[idx] + self.discount_factor * torch.max(self.model(next_state[idx]))

            target_q_values[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not is_done
        self.optimizer.zero_grad()
        loss = self.criterion(target_q_values, predicted_q_values)
        loss.backward()

        self.optimizer.step()
