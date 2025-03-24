# Code used in my Large Thesis
This repository contains the code used in my large thesis. Since I used the Repository (https://github.com/amazon-science/chronos-forecasting/tree/main) as a base this repository contains only modified or additional code.

The BatchScripts directory contains all batch scripts used to run the pretraining and evaluation of all model configurations. Furhthermore it contains the Batch script used to get the Training data.

The ModifiedScripts directory contains the modified train.py to use additional hyperparameter as well as the modified evaluate_new.py to use additional metrics.

Each other directory represents one experiment - speedup, halved training time or one of the detailed searches. In each Directory there are scrips to create the database, insert the configs, run the pretraining and evaluate the models. The results directory in each experiment directory contains the result files with the four used metrics used to evaluate the models. For number of training steps there is no directory as the model versions from the halved training times were used to evaluate this scaling parameter.

To redo an experiment one has to do the following steps:

Install Chronos as preseted on https://github.com/amazon-science/chronos-forecasting/tree/main/scripts
For minimal path adaption store scripts in parent directory of chronos-forecasting
Replace train.py by modified train.py and insert evaluate_new.py in same dir as evaluate.py

Download and convert to Arrow with DataScripts/getTrainingData.py script
    -> once for augmented, once for kernel synth data

Run the DB Script to create the Database

Run the Config Script to generate the configs.
    -> modify paths to downloaded and augmented data in config creation files

run the DB creation, the Configs creation, the Run and Evaluation Script. The scripts have to be executed in the Chronos environment and thus paths have to be adapted to the local environment.
