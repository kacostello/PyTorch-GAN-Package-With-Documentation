# pytorch_GAN_Package

## Requires:

 - Pytorch
 - Matplotlib


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

