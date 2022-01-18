from doctest import OutputChecker
import pytorch_lightning as pl
import torchvision
import torch
from typing import Optional
from .classifier import Classifier
from .loader import OurDataModule
import torchdrift
from matplotlib import pyplot

def corruption_function(x: torch.Tensor):
    return torchdrift.data.functional.gaussian_blur(x, severity=2)

feature_extractor = torchvision.models.resnet18(pretrained=True)
model = Classifier(feature_extractor)
datamodule = OurDataModule()

trainer = pl.Trainer(max_epochs=3, gpus=1, checkpoint_callback=False, logger=False)
trainer.fit(model, datamodule)
trainer.test(model, datamodule=datamodule)



ind_datamodule = datamodule
ood_datamodule = OurDataModule(parent=datamodule, additional_transform=corruption_function)

inputs, _ = next(iter(datamodule.default_dataloader(shuffle=True)))
inputs_ood = corruption_function(inputs)

N = 6
model.eval()
inps = torch.cat([inputs[:N], inputs_ood[:N]])
model.cpu()
predictions = model.predict(inps).max(1).indices

predicted_labels = [["ant","bee"][p] for p in predictions]
pyplot.figure(figsize=(15, 5))
for i in range(2 * N):
    pyplot.subplot(2, N, i + 1)
    pyplot.title(predicted_labels[i])
    pyplot.imshow(inps[i].permute(1, 2, 0))
    pyplot.xticks([])
    pyplot.yticks([])
    
print("drift them")
drift_detector = torchdrift.detectors.KernelMMDDriftDetector()
torchdrift.utils.fit(datamodule.train_dataloader(), feature_extractor, drift_detector)

drift_detection_model = torch.nn.Sequential(
    feature_extractor,
    drift_detector
)

features = feature_extractor(inputs)
score = drift_detector(features)
p_val = drift_detector.compute_p_value(features)
score, p_val