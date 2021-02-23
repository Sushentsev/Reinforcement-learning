import torch
from utils import device


class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl").to(device)
        self.model.eval()

    def act(self, state):
        state = torch.from_numpy(state).to(device)
        with torch.no_grad():
            Q_value_actions = self.model(state)
        action = Q_value_actions.cpu().detach().numpy().argmax()
        return action

    def reset(self):
        pass
