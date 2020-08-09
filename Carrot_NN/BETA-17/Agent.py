from Network import Brain



class Agent:

    def __init__(self):
        self.brain = Brain()
        self.brain.Q = self.brain.modeling_NN()

    def action_process(self, state):
        return self.brain.action_order(state)