# tenas_ccbe
my code for using the NAS method, "TENAS", to discover a better inherently explainable architecture for the task of classifying skin lesions.

This code should be used in conjunction with the code and auxiliary files from https://github.com/CristianoPatricio/coherent-cbe-skin, from https://github.com/VITA-Group/TENAS and from https://github.com/chenwydj/DARTS_evaluation. You should follow their instructions on how to set up your environment.

You should run the following command to apply the TE-NAS method to the skin lesions datasets: python prune_launch.py --space darts --dataset derm7 --gpu 0. This should return you a genotype that you should have to the genotype.py file. 

To train your model, add it to the model_builder.py file inside the modules directory. Next, you should modify model_training.py line 91, where you should add the name of your model. Lastly, you should run the evaluate.py file and change it in line 65 with the name of your model.

