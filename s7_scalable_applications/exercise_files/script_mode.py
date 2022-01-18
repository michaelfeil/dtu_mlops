import torchvision.models as models
import torch
import numpy as np
from torch.utils.benchmark import Timer

model = models.mobilenet_v3_small(pretrained=True)

dummy_input = torch.randn(1, 3,224,224, dtype=torch.float)
dummy_input_batch = torch.randn(128, 3,224,224, dtype=torch.float)


scripted_model = torch.jit.script(model)

print(
    model(dummy_input_batch),
    scripted_model(dummy_input_batch)
)

t0 = Timer(
    stmt='scripted_model(dummy_input_batch)', 
    globals={'scripted_model': scripted_model, "dummy_input_batch": dummy_input_batch})

t1 = Timer(
    stmt='model(dummy_input_batch)',
    globals={'model': model, "dummy_input_batch": dummy_input_batch})

print("approx equal runs",t0.timeit(5), t1.timeit(5))
