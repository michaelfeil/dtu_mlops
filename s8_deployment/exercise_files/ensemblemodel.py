import torch
from torch import nn
from torchvision import models

# TODO: add more

sample_input = torch.randn(1, 3, 224, 224)

class EnsembleModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbones = ["resnet18", "resnet18"]
        
        
        self.backbones = nn.ModuleList(
            [getattr(models, bb)(pretrained=True) for bb in backbones]
        )
        
    def forward(self, x: torch.Tensor):
        res = [bb(x) for bb in self.backbones]
        x =  torch.stack(res, dim=1).sum(dim=1) / len(self.backbones)
        return x

if __name__ == "__main__":
    model = EnsembleModel()
    test_out = model(sample_input)
    script_model = torch.jit.script(model)
    script_model.save('/home/michi/dtu_mlops/s8_deployment/exercise_files/model_store/deployable_model.pt')
