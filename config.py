n_layers = 6
n_head = 6
n_embd = n_head * 64
vocab_size = 65
dropout = 0.2

batch_size = 64
block_size = 256  # context window

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = learning_rate / 10
beta1 = 0.9
beta2 = 0.99
weight_decay = 1e-1


eval_interval = 250
eval_iters = 200
log_interval = 10

dataset = 'shakespeare'

wandb_log = False
wandb_project = 'gpt'
wandb_run_name = 'shakespeare'
