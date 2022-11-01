**PyTorch implementation of the paper Enhancing Representation Learning with Deep Classifiers in Presence of Shortcut**

# Requirements
torch (1.11)

torchvision (0.12)

torchnet (0.0.4)

Pillow

tqdm

# Setting Up
In the global configuration file (*globalconf*), you can set the paths for saving/loading results and models as well as datasets directory.
For the CelebA experiments, due to technical issues in TorchVision, you might need to download the dataset files manually.

The configuration files in the ‘config’ directory are the entries for running the methods (via the *main.py* script), and contain the primary settings/hyperparameters for each algorithm-dataset. 

# Usage Examples

**A. Upstream Training:**

To train the model using the method proposed in the paper (ALR: Adversarial Lens with Random transform) on the CIFAR-10 rotation prediction task biased with the arrow shortcut, run the following in the command line:

python main.py alr_cifar_rot cifar10 arrow arrow --title cifar_arrow_example

The results (trained model and loss curves) will be saved in a directory named ‘cifar_arrow_example’ under the working directory.

Other combinations of the datasets and their modes are:

*cifar10 grad*

*cifar10 clean*

*celeba biased_gender*

*celeba balanced_smile*

 
 
Similarly, to train using the Automatic Shortcut Removal method on the biased CelebA data:

python main.py asr_celeba celeba biased_gender biased_gender --title asr_celeba_example

For the vanilla method, you can use the Spectral Decoupling configs (*sd_celeba*, *sd_cifar_rot*) by changing the logits penalty hyperparameter to zero.

**B. Downstream Training/Evaluation:**

In order to train and test the downstream classifier on the CIFAR-10 arrow biased data, using the representation learned by the model in the first example of A, run:

python main.py dw_cifar cifar10 arrow arrow --feature_extractor_dir ./outputs/cifar_arrow_example --feature_extractor_config alr_cifar_rot

Where the first optional argument is the path to where the upstream trained model (feature extractor) is saved, and the second one is always the configuration used in upstream training. The best accuracy on the test data will be printed after each epoch.

Note that the ‘cifar10’ dataset above refers to the dataset labeled with 10 object classes since used with a downstream configuration. The *dw_cifar* and *dw_celeba* can be used with all the upstream methods except for the Automatic Shortcut Removal, which works with *dw_asr_cifar* and *dw_asr_celeba* . For instance, to obtain the downstream performance in smile detection on CelebA, using the model in the second example of A, run:

python main.py dw_asr_celeba celeba balanced_smile balanced_smile --feature_extractor_dir ./outputs/asr_celeba_example --feature_extractor_config asr_celeba
