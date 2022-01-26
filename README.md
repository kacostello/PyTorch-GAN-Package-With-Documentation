# pytorch_GAN_Package

## Requires:

 - Pytorch
 - Matplotlib

## PyTorch GAN Package General Information
This PyTorch GAN Package includes different GANS using PyTorch and Python programming. The purpose of this package is to give the developer community a useful that will help make and test specific kinds of GANs. The GANs in the package include:

 - Simple GAN
 - Wasserstein GAN
 - Conditional GAN
 - Controllable GAN

This package also includes a Super Trainer file, a ToTrain file, and visualizations. The package is designed so the user does not have to write a GAN entirely from scratch.

If the user has no expreience with GANs, some experience with Python and opaque Machine Learning, or a GAN expert, this tool will be beneficial.

The package design was built with three key features in mind: model building, model training, and model evaluation. For GAN structures that involve novel additions to previous designs, or require the manipulation of technical, subtle changes to the way training is conducted, the package can function as a tool kit for developers and researchers alike.

## Data Input
 - Accept numerical/tabular data: numpy matrix (2d) (rows = observations, columns = attributes)
 - If labeled with a class (0,n), another numpy array (1d) (integer index of the class, ie: 1, 2, 0, 3, ....). The size of this array must equal the number of rows in the matrix above 
 - Accept image data: single channel (grayscale) numpy tensor (3d) (pixels in the x-axis of image, pixels in the y-axis of image, index of images)
 - If labeled with a class (0, n), another numpy array (1d) (integer index of the class, ie: 1, 2, 0, 3, ....). The size of this array must equal the size of the 3rd dimension in the tensor above. 

## Key Fetures for User
Template can be read as: foo(<parameter_name>(required or not){type})
 - Declare(type{string}, gen(optional){pytorch_architecture}, disc(optional){pytorch_architecture}) - Declare an GAN we want to use, and have practitioner-specified architectures if desired. 
 - Train(dataset{numpy_array}, optimizer(optional){pytoroch_optim_object}, drawfromlatentspace(optional){function}, switch_condition(optional){function}, epochs(optional){integer}, lr(optional)={float})
 - Load(type{string}, gen{pytorch_architecture loaded with trained weights}) Assumes gen is a pretrianed pytorch generator, potentially not trained by this package 
 - Save(): saves weight file and the architecture file
 - Sample(n{integer}, class(optional, only applicable for class-generation){integer})
 - Evaluate(flags_for_each_visualization(optional, multiple arguements){boolean}, to_storage(optional){boolean})

