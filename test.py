import torch
import matplotlib.pyplot as plt
from robot_renderer.cartpole_renderer import CartpoleRenderer
renderer = CartpoleRenderer()
images = renderer.render(torch.tensor(
                            [[0., 0.],
                             [0, -1], 
                            [1.0, 1.0]], device="cuda"))
images = images.detach().cpu().numpy()
for i in range(images.shape[0]):
    plt.figure(i)
    plt.imshow(images[i,:,:,:3])
plt.show()