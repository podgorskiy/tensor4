from __future__ import print_function
import numpy as np
import imageio
import torchvision.models as models
import sys
import os
import json
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath('../../'))

import tensor4

use_cuda = torch.cuda.is_available()

use_cuda = False

torch.set_default_tensor_type('torch.FloatTensor')

if use_cuda:
    device = torch.cuda.current_device()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Running on ", torch.cuda.get_device_name(device))


def numpy2torch(x):
    x = torch.from_numpy(x.astype(np.float32))
    if use_cuda:
        return x.cuda()
    else:
        return x.cpu()

def main():
    resnet = models.resnet50(pretrained=True)

    resnet.eval()

    with open("../common/classes.txt", "r") as read_file:
        classes = json.load(read_file)

    classes = {int(k): v for k, v in classes.items()}

    mean = numpy2torch(np.array([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1]))
    std = numpy2torch(np.array([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1]))

    im = imageio.imread('../common/alexnet224x224_input.png')[:, :, :3].transpose((2, 0, 1)) / 255.0
    im = numpy2torch(im).view(1, 3, 224, 224)

    im = (im - mean) / std

    out = tensor4.generate(resnet, args=(im,))
    out = F.softmax(out, 1)
  
    out = out.detach().cpu().view(-1).numpy()

    indices = np.flip(out.argsort(), 0)[:10]
    for i in indices:
        print(str(out[i] * 100.0) + "% " + classes[i])


if __name__ == '__main__':
    main()
