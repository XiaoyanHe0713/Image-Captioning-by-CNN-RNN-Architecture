# PA 3: Experiment Image Captioning with CNN and LSTM Neural Network

Source code from CSE 151B PA3 UCSD Fall 2022

## Contributors (alphabetic order (last, first) by last name):

* Li, Dongze
    * dol005@ucsd.edu
* Long, Yunfan
    * yulong@ucsd.edu
* He, Xiaoyan
    * x6he@ucsd.edu

## Task
In this project, we created captions for a particular image from the data set: COCO 2015 Image Captioning Task by combining CNN and LSTM. To build our training, validation, and test sets, we randomly selected a small portion (1/5) of the original images, each of which contained five representative captions. We used these data sets to train our models using images from the training set, and we used the validation and test sets to calculate the loss, BLEU-1 and BLEU-4 scores. In the end, we were successful in developing an architecture that produces appropriate predicted captions that are similar to the actual captions for the image. We used custom CNN as encoder and LSTM as decoder to build the first model, and we utilized a pretrained Resnet-50 and LSTM to build the second model by transferring the knowledge from Resnet-50. According to our findings, our first model performed better once the hyperparameters hidden size and embedding size were modified. Our first model with custom CNN as encoder at best achieved BLEU-1 and BLEU-4 scores score of 48.502 and 1.986, and cross-entropy test loss of 1.508. Our second model performed better once the optimizer and learning rate were modified. Our second model at best achieved BLEU-1 and BLEU-4 scores of 50.267 and 1.542, and cross-entropy test loss of 2.308.

## How to run

* Run the `get_datasets.ipynb` to download the dataset and extract the features (this must be done on UCSD Datahub). Or you can download the full dataset and modulate the notebook to load the dataset from your local machine.
* For default setting, run `python3 main.py`. Generated captions will be stored in `generated_captions` directory. And all of the expiriments data will be stored in `experiment_data` directory within the `task-1-default-config` directory.
* For different setting, run `python3 main.py config.json` to use the config file. Generated captions will be stored in `generated_captions` directory. And all of the expiriments data will be stored in `experiment_data` directory within the directory of the config file.
* Please do not delete the `generated_captions` directory, otherwise the program will not work.

## Usage

* Define the configuration for your experiment. See `task-1-default-config.json` to see the structure and available options. You are free to modify and restructure the configuration as per your needs.
* Implement factories to return project specific models, datasets based on config. Add more flags as per requirement in the config.
* Implement `experiment.py` based on the project requirements.
* After defining the configuration (say `my_exp.json`) - simply run `python3 main.py my_exp` to start the experiment
* The logs, stats, plots and saved models would be stored in `./experiment_data/my_exp` dir.
* To resume an ongoing experiment, simply run the same command again. It will load the latest stats and models and resume training or evaluate performance.

## Files
- `main.py`: Main driver class
- `experiment.py`: Main experiment class. Initialized based on config - takes care of training, saving stats and plots, logging and resuming experiments.
- `dataset_factory.py`: Factory to build datasets based on config
- `model_factory.py`: Factory to build models based on config
- `file_utils.py`: utility functions for handling files
- `caption_utils.py`: utility functions to generate bleu scores
- `vocab.py`: A simple Vocabulary wrapper
- `coco_dataset.py`: A simple implementation of `torch.utils.data.Dataset` the Coco Dataset
- `get_datasets.ipynb`: A helper notebook to set up the dataset in your workspace

## Help
Make sure to delete the `experiment_data` before running any new experiment. Otherwise, the experiment will resume from the last checkpoint.
