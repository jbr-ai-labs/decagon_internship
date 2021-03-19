PARAMS = {'neg_sample_size': 1,
          'learning_rate': 0.001,
          'epoch': 30,
          'hidden_dim': 64,
          'embed_dim': 32,
          'dropout': 0.1,
          'max_margin': 0.1,
          'batch_size': 512,
          'bias': True,
          'split_ratio': 0.1,
          'seed': 42,
          'loss_type': 'cross_entropy',
          'reduction': 'sum',
          'regularizer': None,
          'num_bases_1': None,
          'num_bases_2': None,
          'early_stopping_window': 2,
          'seed_load': 71,
          'drug_embed_mode': 'mono_se'}
# Model parameters

INPUT_FILE_PATH = "data/input"
# Path to folder with input data.

MODEL_SAVE_PATH = "data/output/saved_model"
# Where to save model during training

SPLIT_SAVE_PATH = "data/split"

MODEL_TO_UPLOAD = None
# Specify model to upload