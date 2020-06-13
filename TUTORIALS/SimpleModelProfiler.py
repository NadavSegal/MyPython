## Pytorch profiler
import torch
import torchvision.models as models

model = models.densenet121(pretrained=True)
x = torch.randn((1, 3, 224, 224), requires_grad=True)

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    model(x)
print(prof)

prof.export_chrome_trace('/home/nadav/Documents/MyPython/TUTORIALS/Tests/profile.json.gz')
prof.export_chrome_trace('/home/nadav/Documents/MyPython/TUTORIALS/Tests/profile.json')
prof.export_chrome_trace('/home/nadav/Documents/MyPython/TUTORIALS/Tests/profile.txt')

## Python profiler
import cProfile
# option 1:
cProfile.run('model(x)')
# option 2:
python -m cProfile myscript.py
python -m cProfile -o output.file mine.py <args>