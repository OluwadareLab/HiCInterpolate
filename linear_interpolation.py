import torch








batch_size, channels, h, w = 32, 1, 64, 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
I_0 = torch.rand((batch_size, channels, h, w), device=device)
I_2 = torch.rand((batch_size, channels, h, w), device=device)
t = 0.5
pred_linear = (1.0 - t) * I_0 + t * I_2