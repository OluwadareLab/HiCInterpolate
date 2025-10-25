import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from torch import optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import warnings
from torchvision.utils import save_image

warnings.simplefilter('ignore')


def gram(tensor):
    return torch.mm(tensor, tensor.t())


def gram_loss(noise_img_gram, style_img_gram, N, M):
    return torch.sum(torch.pow(noise_img_gram - style_img_gram, 2)).div((np.power(N*M*2, 2, dtype=np.float64)))


cont_img = Image.open('./content_img_1.jpg')
style_img = Image.open('./style_img.jpeg')

transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])])

content_image = transform(cont_img).unsqueeze(0).cuda()
style_image = transform(style_img).unsqueeze(0).cuda()


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.vgg = models.vgg19(pretrained=True).features

    def get_content_activations(self, x):
        return self.vgg[:32](x)

    def get_style_activations(self, x):
        return [self.vgg[:30](x)] + [self.vgg[:21](x)] + [self.vgg[:12](x)] + [self.vgg[:7](x)] + [self.vgg[:4](x)]

    def forward(self, x):
        return self.vgg(x)


vgg = VGG().cuda().eval()

for param in vgg.parameters():
    param.requires_grad = False

content_activations = vgg.get_content_activations(content_image).detach()
style_activations = vgg.get_style_activations(style_image)
content_F = content_activations.view(512, -1)

for i in range(len(style_activations)):
    style_activations[i] = style_activations[i].squeeze().view(
        style_activations[i].shape[1], -1).detach()

gram_matrices = [gram(style_activations[i])
                 for i in range(len(style_activations))]

noise = torch.randn(1, 3, 224, 224, device='cuda', requires_grad=True)
adam = optim.Adam(params=[noise], lr=0.01)

for iteration in range(10000):
    adam.zero_grad()
    noise_content_activations = vgg.get_content_activations(noise)
    noise_content_F = noise_content_activations.view(512, -1)
    content_loss = 1/2. * torch.sum(torch.pow(noise_content_F - content_F, 2))
    noise_style_activations = vgg.get_style_activations(noise)
    for i in range(len(noise_style_activations)):
        noise_style_activations[i] = noise_style_activations[i].squeeze().view(
            noise_style_activations[i].shape[1], -1)

    noise_gram_matrices = [gram(noise_style_activations[i])
                           for i in range(len(noise_style_activations))]
    style_loss = 0
    for i in range(len(style_activations)):
        N, M = noise_style_activations[i].shape[0], noise_style_activations[i].shape[1]
        style_loss += (gram_loss(noise_gram_matrices[i],
                       gram_matrices[i], N, M) / 5.)

    style_loss = style_loss.cuda()
    total_loss = content_loss + 10000 * style_loss

    if iteration % 1000 == 0:
        print("Iteration: {}, Content Loss: {}, Style Loss: {}".format(
            iteration, content_loss.item(), 10000 * style_loss.item()))
        save_image(noise, filename='./generated/iter_{}.png'.format(iteration))

    total_loss.backward()
    adam.step()
