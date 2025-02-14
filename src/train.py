import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from model import SmallLanguageModel, ModelConfig  # Remove src. prefix
import random
# ... rest of the code ... 