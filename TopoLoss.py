# Pls use ```pip install torch_topological``` to install some neccessary libs.

import torch_topological.nn as ttnn
class TopoLoss(nn.Module):
    def __init__(self, auto_scale=True):
        super(TopoLoss, self).__init__()
        self.auto_scale = auto_scale
    # @timeit
    def forward(self, inputs, target):
        persistence_layer = ttnn.CubicalComplex()
        before_persistence_diagram = persistence_layer(inputs)
        after_persistence_diagram = persistence_layer(target)
        distance = ttnn.WassersteinDistance()(before_persistence_diagram[0][0], after_persistence_diagram[0][0])
        # scalar
        # Use Algorithms like PPO to make sure the stability of your training. If not, delete ```self.PPO```
        if self.PPO:
            distance = self.PPO(distance)
        return distance
