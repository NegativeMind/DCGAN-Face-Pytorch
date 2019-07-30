# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import pylab

import torch.backends.cudnn as cudnn

from generator import Generator
from discriminator import Discriminator


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':

    manualSeed = 999 # Set random seem for reproducibility
    #manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    save_dir = 'save'

    dataroot = './data/celeba' # Root directory for dataset (subdirectory:img_align_celeba)
    # Celeb-A Faces dataset http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    # https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg

    workers = 2 # Number of workers for dataloader
    batch_size = 128 # Batch size during training
    image_size = 64 # Spatial size of training images. All images will be resized to this size using a transformer.

    nc = 3 # Number of channels in the training images. For color images this is 3
    nz = 100 # Size of z latent vector (i.e. size of generator input)
    ngf = 64 # Size of feature maps in generator
    ndf = 64 # Size of feature maps in discriminator
    ngpu = 1 # Number of GPUs available. Use 0 for CPU mode.

    num_epochs = 90 # Number of training epochs
    lr = 0.0002 # Learning rate for optimizers
    beta1 = 0.5 # Beta1 hyperparam for Adam optimizers
    

    # Create the dataset
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    if (device.type == 'cuda'):
        cudnn.benchmark = True

    
    # Create the generator
    netG = Generator(ngpu, nc, nz, ngf).to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    netG.apply(weights_init)
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


    # Create the Discriminator
    netD = Discriminator(ngpu, nc, ndf).to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    netD.apply(weights_init)
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))


    # We will use the Binary Cross Entropy loss
    criterion = nn.BCELoss() # Initialize BCELoss function

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # Lists to keep track of progress
    d_losses = np.zeros(num_epochs)
    g_losses = np.zeros(num_epochs)
    real_scores = np.zeros(num_epochs)
    fake_scores = np.zeros(num_epochs)

    iters = 0

    # Training Loop
    print("Starting Training Loop...")
    for epoch in range(num_epochs):# For each epoch

        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
        
            #############################################################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            #############################################################
            
            ## Train with all-real batch
            netD.zero_grad()

            real_cpu = data[0].to(device)# Format batch
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            
            output = netD(real_cpu).view(-1)# Forward pass real batch through D
            errD_real = criterion(output, label)# Calculate loss on all-real batch
            errD_real.backward()# Calculate gradients for D in backward pass
            D_x = output.mean().item()


            ## Train with all-fake batch
            noise = torch.randn(b_size, nz, 1, 1, device=device)# Generate batch of latent vectors
            fake = netG(noise)# Generate fake image batch with G
            label.fill_(fake_label)
            
            output = netD(fake.detach()).view(-1)# Classify all fake batch with D
            errD_fake = criterion(output, label)# Calculate D's loss on the all-fake batch
            errD_fake.backward()# Calculate the gradients for this batch
            D_G_z1 = output.mean().item()
            
            errD = errD_real + errD_fake# Add the gradients from the all-real and all-fake batches
            optimizerD.step()# Update D


            #############################################################
            # (2) Update G network: maximize log(D(G(z)))
            #############################################################

            netG.zero_grad()

            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            errG = criterion(output, label)# Calculate G's loss based on this output
            errG.backward()# Calculate gradients for G
            D_G_z2 = output.mean().item()
            optimizerG.step()# Update G
        

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            d_losses[epoch] = d_losses[epoch] * (i/(i+1.)) + errD.item() * (1./(i+1.))
            g_losses[epoch] = g_losses[epoch] * (i/(i+1.)) + errG.item() * (1./(i+1.))

            real_scores[epoch] = real_scores[epoch] * (i/(i+1.)) + D_x * (1./(i+1.))
            fake_scores[epoch] = fake_scores[epoch] * (i/(i+1.)) + D_G_z1 * (1./(i+1.))
        
            # Check how the generator is doing by saving G's output on fixed_noise
            generated_image = []
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
                generated_image.append(vutils.make_grid(fake, padding=2, normalize=True))

            # Plot the fake images from the last epoch
            plt.figure(figsize=(10, 10))
            plt.title("Fake Images")
            plt.imshow(np.transpose(generated_image[-1], (1, 2, 0)))
            plt.savefig('./generated/generate_' + str(iters) + '.png')
            plt.close()

            iters += 1

        # Save and plot Statistics
        np.save(os.path.join(save_dir, 'd_losses.npy'), d_losses)
        np.save(os.path.join(save_dir, 'g_losses.npy'), g_losses)
        np.save(os.path.join(save_dir, 'fake_scores.npy'), fake_scores)
        np.save(os.path.join(save_dir, 'real_scores.npy'), real_scores)
    
        plt.figure()
        pylab.xlim(0, num_epochs + 1)
        plt.plot(range(1, num_epochs + 1), d_losses, label='d loss')
        plt.plot(range(1, num_epochs + 1), g_losses, label='g loss')    
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'loss.pdf'))
        plt.close()

        plt.figure()
        pylab.xlim(0, num_epochs + 1)
        pylab.ylim(0, 1)
        plt.plot(range(1, num_epochs + 1), fake_scores, label='fake score')
        plt.plot(range(1, num_epochs + 1), real_scores, label='real score')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'accuracy.pdf'))
        plt.close()


    # Save models
    torch.save(netG.state_dict(), "./model/generator.pt")
    torch.save(netD.state_dict(), "./model/discriminator.pt")
