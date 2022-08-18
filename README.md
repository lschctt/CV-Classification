# CV-Classification
##  **Summary:**

There are some neutral networks of computer vision, such as GoogleNet, ResNet, DenseNet. They have achieved great results on many datasets and tasks. In order to have a better command of these networks and study to get a good result by adjusting parameters, I try to train and test them at some classical datasets. For each model, you have many methods to optimize and you can also choose pre-trained networks. Welcome to have a try !

##  **Dataset:**

### MNIST:

MNIST dataset consist of gray images of handwritten numbers by different people, which with size of $28\times28\times1$​. MNIST consists of images of 10 classes, i.e. number from 0 to 9. The training and test sets contain 60,000 and 10,000 images respectively.

### CIFAR: 

The two CIFAR datasets consist of colored natural images with size of $32\times32\times3$. CIFAR-10 consists of images of 10 classes and CIFAR-100 consists of images of 100 classes. The training and test sets contain 50,000 and 10,000 images respectively.

## Code Base Structure

- **DataLoader.py:** get dataset and datasetLoader for loading specific data
- **define:** Contains definition of model, optimizer,  scheduler and so on
- **get_train_test:** Contains the definitions for "train" and "test".
- **options.py:** Contains all the options for the argparser.
- **other_functions:** Some functions of get_dim, get_channel and so on.
- **GoogleNet/ResNet/DenseNet.py:** Contains the definitions for specific model and pre-trained model

## Some results

### pre-trained densenet121 on CIFAR-10:

<img src="pictures of readme\预训练densenet121_224x224.png" alt="预训练densenet121_224x224" style="zoom:40%;" />

### ResNet[10,10,10] on CIFAR-10:

<img src="pictures of readme\ResNet_n[10, 10, 10]_随机翻转和标准化.png" alt="ResNet_n[10, 10, 10]" style="zoom:40%;" />
