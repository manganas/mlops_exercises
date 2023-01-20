import torch

input = torch.randn(10, 3, 50, 50)

scale = torch.tensor(0.5)
zero_point = torch.tensor(0.1)
quant_tensor = torch.quantize_per_tensor(input, scale, zero_point, torch.int16)

# print(quant_tensor)

unquantized_tensor = quant_tensor.dequantize()
