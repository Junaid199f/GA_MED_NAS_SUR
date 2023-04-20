import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict
import math
from torchvision import transforms
class Augmentation:
  def get_augmentation(self):
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform