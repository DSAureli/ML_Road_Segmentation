# ML Project 2 - Road Segmentation

Repository containing the code for the [Project 2 - Road Segmentation](https://github.com/epfml/ML_course/blob/master/projects/project2/project2_description.pdf) of the [Machine Learning course](https://www.epfl.ch/labs/mlo/machine-learning-cs-433/) at EPFL.

The team "Epok" is composed by:

- Davide Nanni ([davide.nanni@epfl.ch](mailto:davide.nanni@epfl.ch))
- Nils Frédérik Heiden Olsen ([nils.olsen@epfl.ch](mailto:nils.olsen@epfl.ch))
- Paul Victor Augustin Mermod ([paul.mermod@epfl.ch](mailto:paul.mermod@epfl.ch))

## Setup

The needed packages names, as well as their version, are contained in the requirements.txt file. They can be automatically installed with the command:

```shell
pip install -U -r requirements.txt
```

(you may need to use ```sudo``` on Linux, or alternatively pass the ```--user``` parameter to install them locally)

The training/validation/test datasets and the pre-trained weights are provided separately at the following [link](https://drive.google.com/file/d/1B4PD1NiZIQo1idK1-V6F9uMKFbjGUK6I/view?usp=sharing). The zip file must be put as-is in the main directory, together with all source files. In the end, you should have the following files together in the same folder:

```
gen_sub.py
image_augmentation.py
mask_to_submission.py
ml_p2_data.zip
run.py
unet.py
```

## Structure

- ```image_augmentation.py``` : contains the augmentation generator class ```AugmentedSequence```, subclass of the Keras ```Sequence``` class, which generates batches of augmented images by applying the provided ```albumentations``` composition. Refer to the class docstring for more info. You can test the class functioning by directly executing the file, but in that case you'll need to organize
data as the following structure:

```
image_augmentation.py
training
├── images (satellite images)
└── groundtruth (relative masks)
```

- ```unet.py``` : contains the actual code for initializing and training the model. If directly executed, it calls the function ```train``` without providing custom arguments. For more information, please refer to the function docstring. The default required tree structure is as follows:

```
image_augmentation.py
unet.py
data
├── training
│   ├── image
│   └── mask
└── validation
    ├── image
    └── mask
```

- ```gen_sub.py``` : contains the prediction and submission generation. If directly executed, it calls the function ```generate_submission```, which expects the weight file ```./last_best_model.h5``` and the directory ```./test_set_images/```, containing the test images, available at the source of execution and generates the submission file ```submission.csv``` calling the appropriate function in ```mask_to_submission.py```. Refer to the function docstring for more information.

- ```mask_to_submission.py``` : the provided file for generating the submission csv from predicted outputs. Used by ```gen_sub.py``` for generating the submission file.

- ```run.py``` : contains the whole pipeline execution for reproducing the results. More info on usage hereunder.


## Usage

In order to reproduce all the necessary steps for the final submission, it is sufficient to run the ```run.py``` file:

``` shell
python run.py
```

This will take care of unzipping the necessary data, creating the required directories, training the model, making the predictions on the test set (output in the directory ```out```) and producing the submission file ```submission.csv```.

We also provide the saved weights of the pre-trained model, so that you don't have to run the training yourself. You can skip the training passing the ```-l``` argument to the script:

``` shell
python run.py -l
```

## Results

The best submission on AICrowd (#28949) was obtained by the user DavideNanni on Sat, 14 Dec 2019, 12:32:14, with a F1 score of 0.912 and a secondary score of 0.953. The training was done on a 6 cores @2.60GHz CPU in ~19.5 hours.

## Reproducibility

The first lines in the file ```run.py``` take care of seeding all sources of randomness for the used libraries and disabling the GPU availability to Tensorflow. The former was done for having reproducible results, the latter both for reproducibility (CUDA ML libraries introduce randomness) and for avoiding crashes due to limits in graphics memory available. Even though, it may happen that the obtained results for the same training procedure slightly differ from the ones we obtained with the same code, as some operation-level seeds can't be set for the used model.