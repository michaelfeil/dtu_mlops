import torchvision.models as models
import torch
import numpy as np
from ptflops import get_model_complexity_info


def inference(model):


    device = torch.device("cuda")
    
    
    model.to(device)
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                                print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    dummy_input = torch.randn(1, 3,224,224, dtype=torch.float).to(device)
    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings=np.zeros((repetitions,1))
    #GPU-WARM-UP
    for _ in range(10):
        # as in this blog post
        # https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwi7md-xu7n1AhVaS_EDHdLzCFgQFnoECAoQAQ&url=https%3A%2F%2Fdeci.ai%2Fresources%2Fblog%2Fmeasure-inference-time-deep-neural-networks%2F&usg=AOvVaw15nc912etEY_L0ecRj1JiO
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(f"{model}: mean is: {mean_syn}")
    return macs


if __name__ == "__main__":
    model = models.resnet18(pretrained=True)
    macs_resnet18 = inference(model)
    
    mobilenet_v3_small = models.mobilenet_v3_small(pretrained=True)
    macs_resnet_v3small = inference(mobilenet_v3_small)
    
    print("resnet/mobilenetv3", float(macs_resnet18.split(" GMac")[0])/float(macs_resnet_v3small.split(" GMac")[0]))