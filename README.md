
Stanford CS231n 2024 assignments, plus style transfer from 2020, and replication of ResNet for CIFAR-10.

## Assignment 1

K nearest neighbour. distance matrix, using cross validation to find the best k.

SVM classifier. CIFAR-10 image preprocessing. Linear SVM forward and backward functions. 

Softmax forward and backward.

Two layer MLP network.

## Assignment 2

Feed forward network. Optimization methods: SGD, SGD + momentum, RMSProp, ADAM.

Convolutional network.

Batch normalization. Layer normalization. 

Dropout.

Pytorch exercises. 

## Assignment 3

### Image Captioning
#### Coco Dataset for Captioning

Images features, dim=4096, are extracted from the fc7 layer of the VGG-16 network pretrained on ImageNet (not COCO).
And further, a dimensionality reduced version with dim=512 is generated by PCA.

Word to integer index and integer index to word mappings are stored in a JSON file.
4 special tokes are added, `<START>`, `<END>`, `<UNK>` for out-of-vocab words, and `<NULL>` for padding.


#### RNN and LSTM Captioning

Vanilla RNN and LSTM are implemented using purely Numpy.

The image representation extracted from VGG and after a linear projection, is used as initial hidden state $h_0$ for RNN/LSTM.

The experiments overfit 50 training data points from COCO. 

#### Transformer Captioning

The transformer for image captioning is decoder only. Because the image features are used as output of encoder, and are attended at every decoder layer (block).
Each decoder block has 4 layers, a self-attention layer, a multi-head attention layer (to attend image features), followed by two linear layers. 

At training time, for each caption x, `x[:-1]` is used as input, and `x[1:]` is the target. `x[:-1]` goes through an embedding layer and positional embedding layer before feeding into the first decoder layer.
The captions in a batch are packed with the token `<NULL>` so they have the same length. The `<NULL>` tokens do not contribute to loss and as a result no gradients.

At test time (sampling), the first input is the `<START>` token, and tokens generated at each step are concatenated together and fed back as input until we have reached the `max_len` number of tokens or `<END>` is produced. 

### GAN

The objective function $\displaystyle \min_G \max_D E_{x \sim p_{data}}[logD(x)] + E_{z \sim p(z)}[log(1 - D(G(z)))]$ 
works well with Discriminator D but has a problem in optimizing Generator G.
At the start of training, $G(z)$ generates roughly random noise, therefore $D(G(z))$ is close to 0. This means  $-log(1 - D(G(z)))$ is close to 0 too, and the gradients propagated back to generator $G$ is very small.
But if we use objective function  $\displaystyle \max_GE_{z \sim p(z)}[log(D(G(z)))]$, when $D(G(z))$ is close to 0, $-log(D(G(z)))$ is a big number, which generates strong gradients back to G.

Least squares GAN found the vanilla GAN's sigmoid cross-entropy loss function may lead to the vanishing gradients problem during the learning process. The paper proposed to use least square loss function, and showed that it can stablize training and generate better images. The discriminator optimizer pushes $D(x)$ to 1 where $x \sim p_{data}$ and $D(G(z))$ to zero, whereas the generator optimizer wants to push $D(G(z))$ to 1.

### Self-supervised Learning

Self-supervised learning allows a model to learn and generate a "good" representation for images without labels. "Good" means images in the dataset representing **semantically** similar entities should have similar representations, and different images should have different representations.

<img src="https://production-media.paperswithcode.com/methods/Screen_Shot_2020-07-02_at_4.31.34_PM_7zlWDQE.png" width="500">

The authors cleverly constructed "labels" by random data augmentation. Specifically, given an image $x$, SimCLR uses <ins>two different data augmentation schemes</ins> $t$ and $t'$ to generate the <ins>positive pair of images</ins> $\tilde{x}_i$ and $\tilde{x}_j$. $f$ is a basic encoder net that extracts representation vectors $h_i$ and $h_j$ respectively from the augmented data samples. Finally, a small neural network projection head $g$ maps the representation vectors to the space where the contrastive loss is applied. The goal of the contrastive loss is to **maximize agreement between the final vectors** $z_i = g(h_i)$ and $z_j = g(h_j)$:

$$
l(i, j) = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau) }{\sum_{k=1}^{2N} \mathbb{1}_{k \neq i} \exp (\text{sim}(z_i, z_k)/ \tau) }
$$

where $\mathbb{1} \in \\{0, 1\\}$ is an indicator function that outputs 1 if $k\neq i$ and 0 otherwise. $\tau$ is a temperature hyperparameter that determines how fast the exponentials increase.   
$sim(z_i, z_j) = \frac{z_i \cdot z_j}{||z_i||||z_j||}$ 

is the (normalized) dot product between vectors $z_i$ and $z_j$.  
  
The loss function is designed that it can not only push positive pairs closer but also push negative pairs apart.

## Assignments from other years

### style transfer

**Idea**: generate an image that reflects the content of one image and the style of another by incorporating both in our loss function. We can then use this hybrid loss function to perform gradient descent **not on the parameters** of the model, **but instead on the pixel values** of our original image.

#### Loss
The loss function is a weighted sum of three terms: content loss + style loss + total variation loss.

##### content loss
The content loss is given by, for a given layer $\ell$:
$L_c = w_c \times \sum_{i,j} (F_{ij}^{\ell} - P_{ij}^{\ell})^2$

where:
* $F^\ell \in \mathbb{R}^{C_\ell \times M_\ell}$ is the feature map of the current generated image
* $P^\ell \in \mathbb{R}^{C_\ell \times M_\ell}$ is the feature map of the content source image
* $C_\ell$ is the number of filters/channels in layer $\ell$, and $M_\ell=H_\ell\times W_\ell$ is the number of elements in each feature map.
* $w_c$ is the weight of the content loss

##### style loss
The style loss uses Gram matrix which is an approximation to the covariance matrix -- we want the **activation statistics of our generated image to match the activation statistics of our style image**, and matching the (approximate) covariance is one way to do that. 

Given a feature map $F^\ell$ of shape $(C_\ell, M_\ell)$, the Gram matrix has shape $(C_\ell, C_\ell)$, ie., $G_{ij}^{\ell}=\sum_k F_{ik}^{\ell}F_{jk}^{\ell}$.

For a given layer $\ell$, the style loss is:
$L_s^\ell = w_\ell \sum_{i, j} \left(G^\ell_{ij} - A^\ell_{ij}\right)^2$

where:
* $G^\ell$ is the Gram matrix from the feature map of the current generated image
* $A^\ell$ is the Gram Matrix from the feature map of the source style image
* $w_\ell$ is the weight of the style loss

In practice we usually compute the style loss at a set of layers, rather than just a single layer. In this case, just sum up the style loss of all the layers concerned.

##### Total variation loss
The total variation loss is to encourage smoothness in the image. 

The "total variation" is computed as the sum of the squares of differences in the pixel values for all pairs of pixels that are next to each other (horizontally or vertically):

$L_{tv} = w_t \times \left(\sum_{c=1}^3\sum_{i=1}^{H-1}\sum_{j=1}^{W} (x_{i+1,j,c} - x_{i,j,c})^2 + \sum_{c=1}^3\sum_{i=1}^{H}\sum_{j=1}^{W - 1} (x_{i,j+1,c} - x_{i,j,c})^2\right)$

Since we only compute total variation loss of the generated image, $C$ is always 3 for a RGB image.

#### Training
* extract features of the content and style images using a trained ConvNet. 
* initialize the generated image as random noise or just a copy of the original content image. Set `image.required_grad_(True)`.
* At each training iteration:
  * extract features of the generated image
  * compute loss as described above
  * propagate back gradients into the image tensor to update it
  
That's it.


## Resnet for CIFAR-10

I implemented a Resnet model for CIFAR-10 by following the [original resnet paper](https://arxiv.org/abs/1512.03385). The implementation strictly follows section 4.2 of the paper. 

Specifically, the input layer is a 3 x 3 convolutional layer with 16 filters. Then the model has a stack of 6n layers (n = 3 in this case). All conv layers are 3 x 3 convolutions on the feature map sizes {32, 16, 8} respectively, with 2n layers for each feature map size. The number of filters are {16, 32, 64} respectively. 
The residual connection uses option A described in the paper, i.e. use stride=2 Max pooling, and pad zeros for increased dimensions.
The last is a feedforward layer with 10 output classes.

So in total there are 20 layers, with 269914 (0.27 MB) parameters.
The validation and test accuracy on CIFAR-10 are between 91.5-92.0 %, consistent with the paper.

It is pretty cool that such a small network has so good performance.

<img src="resnet/asset/resnet_training.png" alt="resnet training acc" width="550"/>

See [Wandb training log here](https://wandb.ai/dalucheng/restnet-cifar10/runs/wjjktvta). 