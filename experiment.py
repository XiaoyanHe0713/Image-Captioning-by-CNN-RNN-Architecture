################################################################################
# CSE 151B: Programming Assignment 3
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin, Yash, James
# Fall 2022
################################################################################

import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
import math
from tqdm import tqdm
from copy import deepcopy
import caption_utils
import pandas as pd
from os import listdir
from PIL import Image as PImage
import torchvision.transforms as transforms

TEST_IDS_PATH = './test_ids.csv'
ROOT_STATS_DIR = './experiment_data'
from dataset_factory import get_datasets
from file_utils import *
from caption_utils import *
from model_factory import get_model
import coco_dataset as coda

device = torch.device("cuda")

# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        self.__coco, self.__coco_test, self.__vocab, self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(
            config_data)

        # Setup Experiment
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.resize = transforms.Compose([
            transforms.Resize(config_data['dataset']['img_size'], interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(config_data['dataset']['img_size'])
        ])
        self.__epochs = config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__early_stop = config_data['experiment']['early_stop']
        self.__patience = config_data['experiment']['patience']
        self.__batch_size = config_data['dataset']['batch_size']
        self.__windows = config_data['experiment']['windows']
        self.__length = config_data['generation']['max_length']
        self.__goodids = config_data['generation']['good_ids']
        self.__badids = config_data['generation']['bad_ids']

        # Init Model
        self.__model = get_model(config_data, self.__vocab)
        self.__best_model = deepcopy(self.__model.state_dict())

        # criterion
        # TODO
        self.__criterion = torch.nn.CrossEntropyLoss()

        # optimizer
        # TODO
        if config_data['experiment']['optimizer'] == 'Adam':
            self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr=config_data['experiment']['learning_rate'])
        else:
            self.__optimizer = torch.optim.SGD(self.__model.parameters(), lr=config_data['experiment']['learning_rate'])        

        # LR Scheduler
        # TODO
        self.__lr_scheduler = torch.optim.lr_scheduler.StepLR(self.__optimizer, step_size=1, gamma=0.9)

        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])
        else:
            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.to(device).float()
            self.__criterion = self.__criterion.to(device)

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        start_epoch = self.__current_epoch
        patience_count = 0
        min_loss = 100
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            print(f'Epoch {epoch + 1}')
            print('--------')
            start_time = datetime.now()
            self.__current_epoch = epoch
            print('Training...')
            print('-----------')
            train_loss = self.__train()
            print('Validating...')
            print('-------------')
            val_loss = self.__val()

            # save best model
            if val_loss < min_loss:
                min_loss = val_loss
                self.__best_model = deepcopy(self.__model.state_dict())

            # early stop if model starts overfitting
            if self.__early_stop:
                if epoch > 0 and val_loss > self.__val_losses[epoch - 1]:
                    patience_count += 1
                if patience_count >= self.__patience:
                    print('\nEarly stopping!')
                    break

            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()
            if self.__lr_scheduler is not None:
                self.__lr_scheduler.step()
        self.__model.load_state_dict(self.__best_model)

    def __compute_loss(self, images, captions, teacher_forcing=False):
        """
        Computes the loss after a forward pass through the model
        """
        # TODO

        # run forward and compute loss
        _, loss = self.__model(images, captions, teacher_forcing = teacher_forcing, learning = False)

        return loss

    def __train(self):
        """
        Trains the model for one epoch using teacher forcing and minibatch stochastic gradient descent
        """
        
        # TODO
        losses = []
        for i, sample in enumerate(tqdm(self.__train_loader)):
            # get the inputs
            images, captions = sample[0], sample[1]
            images = images.to(device)
            captions = captions.to(device)
            
            # train model
            _, loss = self.__model(images, captions, teacher_forcing=True, criterion=self.__criterion,
                                   optimizer=self.__optimizer)
            losses.append(loss.item())

        return np.mean(losses)

    def __generate_captions(self, img_id, deterministic=False, save_good=True, save_bad=True):
        """
        Generate captions without teacher forcing
        Params:
            img_id: Image Id for which caption is being generated
            testing: whether the image_id comes from the validation or test set
        Returns:
            tuple (list of original captions, predicted caption)
        """
        
        # TODO
        # get original captions
        original_captions = []
        for ann in self.__coco_test.imgToAnns[img_id]:
            original_captions.append(ann['caption'].lower().replace(".", " ."))
        
        # get image
        img_path = self.__coco_test.loadImgs(img_id)[0]['file_name']

        root_test = os.path.join("./data/images/", 'test')

        img = PImage.open(os.path.join(root_test, img_path)).convert('RGB')

        # get generated caption
        img_torch = self.resize(img)
        img_torch = self.normalize(np.asarray(img_torch))
        # add a batch dimension
        img_torch = img_torch.unsqueeze(0)
        captions = torch.zeros((1, self.__length), dtype=torch.long).to(device)
        predicted_caption, _ = self.__model(img_torch, captions, deterministic=deterministic, learning=False, teacher_forcing=False)
        pred_str = ""

        predicted_caption = predicted_caption.flatten()
        predicted_caption = predicted_caption.tolist()
        for pred in predicted_caption:
            word = self.__vocab.idx2word[pred]
            if word == "<start>":
                continue
            elif word == "<end>" or word == "<pad>":
                break
            else:
                pred_str += word + " "

        # splt caption and predicted caption into list of words
        original_captions_split = []
        for i in range(len(original_captions)):
            original_captions_split.append(original_captions[i].split())
        pred_str_split = pred_str.split()

        # compute bleu score
        B1 = bleu1(original_captions_split, pred_str_split)
        B4 = bleu4(original_captions_split, pred_str_split)

        good_count = 0
        bad_count = 0
        if B1 >= 80 and save_good:
            good_count = 1

            self.__save_generated_captions(img, img_id, original_captions, pred_str, B1, B4)

        elif B1 <= 10 and save_bad:
            bad_count = 1

            self.__save_generated_captions(img, img_id, original_captions, pred_str, B1, B4)

        return B1, B4, good_count, bad_count, original_captions, pred_str

    def __save_generated_captions(self, img, img_id, original_captions, predicted_caption, B1, B4):
        # print and save image
            fig = plt.figure()
            plt.axis('off')
            plt.imshow(img)
            save_path = os.path.join('./generated_captions', str(img_id) + '.png')

            if self.__windows:
                save_path = save_path.replace('\\', '/')

            plt.savefig(save_path)
            plt.close(fig)

            # save predicted caption and original captions
            with open(os.path.join('./generated_captions', str(img_id) + '_caption.txt'), 'w') as f:
                f.write("Original Captions:\n")
                for caption in original_captions:
                    f.write(caption + "\n")
                f.write("\nPredicted Caption:\n")
                f.write(predicted_caption + "\n")

            # save bleu score
            with open(os.path.join('./generated_captions', str(img_id) + '_bleu.txt'), 'w') as f:
                f.write("Bleu-1 Score: " + str(B1) + "\n")
                f.write("Bleu-4 Score: " + str(B4) + "\n")

    def __str_captions(self, img_id, predicted_caption):
        """
            !OPTIONAL UTILITY FUNCTION!
            Create a string for logging ground truth and predicted captions for given img_id
        """
        result_str = "Captions: Img ID: {},\nPredicted: {}\n".format(
            img_id, predicted_caption)
        return result_str

    def __val(self):
        """
        Validate the model for one epoch using teacher forcing
        """

        # TODO
        with torch.no_grad():
            losses = []
            # one epoch of validation using teacher forcing
            for i, sample in enumerate(tqdm(self.__val_loader)):
                images = sample[0].to(device)
                captions = sample[1].to(device)

                # forward pass
                _, loss = self.__model(images, captions, teacher_forcing=True, learning=False, criterion=self.__criterion, 
                                    optimizer=self.__optimizer)
                losses.append(loss.item())

            return np.mean(losses)

    def test(self, caption_experiment=False):
        """
        Test the best model on test data. Generate captions and calculate bleu scores
        """
        with torch.no_grad():
            # TODO
            losses = []
            for i, sample in enumerate(tqdm(self.__test_loader)):
                images = sample[0].to(device)
                captions = sample[1].to(device)

                # forward pass
                _, loss = self.__model(images, captions, teacher_forcing=True, learning=False, criterion=self.__criterion, 
                                    optimizer=self.__optimizer)
                losses.append(loss.item())
            final_loss = np.mean(losses)

            print('Test loss: ', final_loss)

        # save test loss
        write_to_file_in_dir(self.__experiment_dir, 'test_loss.txt', str(final_loss))

        with open('test_ids.csv', 'r') as f:
            test_ids = f.read().replace('\"', '').split(',')
        test_ids = [int(id) for id in test_ids]

        # calculate bleu scores
        bleu1_scores = []
        bleu4_scores = []
        good_count = 0
        bad_count = 0
        save_good = False
        save_bad = False

        for img_id in tqdm(test_ids):
            if good_count < 10:
                save_good = True
            else:
                save_good = False
            if bad_count < 10:
                save_bad = True
            else:
                save_bad = False

            B1, B4, good, bad, _, _ = self.__generate_captions(img_id, deterministic=False, save_good=save_good, save_bad=save_bad)

            if B1 >= 80 and save_good:
                self.__goodids.append(img_id)
            
            if B1 <= 10 and save_bad:
                self.__badids.append(img_id)
                
            bleu1_scores.append(B1)
            bleu4_scores.append(B4)
            good_count += good
            bad_count += bad

        print('Bleu-1 Score: ', np.mean(bleu1_scores))
        print('Bleu-4 Score: ', np.mean(bleu4_scores))


        if caption_experiment:
            for img_id in tqdm(self.__goodids):
                good_captions = []
                self.__model.temp = 0.4
                _, _, _, _, orginial_captions, predicted_caption = self.__generate_captions(img_id, deterministic=False, save_good=False, save_bad=False)
                good_captions.append(predicted_caption)
                self.__model.temp = 5.0
                _, _, _, _, orginial_captions, predicted_caption = self.__generate_captions(img_id, deterministic=False, save_good=False, save_bad=False)
                good_captions.append(predicted_caption)
                self.__model.temp = 0.0001
                _, _, _, _, orginial_captions, predicted_caption = self.__generate_captions(img_id, deterministic=False, save_good=False, save_bad=False)
                good_captions.append(predicted_caption)
                _, _, _, _, orginial_captions, predicted_caption = self.__generate_captions(img_id, deterministic=True, save_good=False, save_bad=False)
                good_captions.append(predicted_caption)
                self.__save_captions(img_id, good_captions)

            for img_id in tqdm(self.__badids):
                bad_captions = []
                self.__model.temp = 0.4
                _, _, _, _, orginial_captions, predicted_caption = self.__generate_captions(img_id, deterministic=False, save_good=False, save_bad=False)
                bad_captions.append(predicted_caption)
                self.__model.temp = 5.0
                _, _, _, _, orginial_captions, predicted_caption = self.__generate_captions(img_id, deterministic=False, save_good=False, save_bad=False)
                bad_captions.append(predicted_caption)
                self.__model.temp = 0.0001
                _, _, _, _, orginial_captions, predicted_caption = self.__generate_captions(img_id, deterministic=False, save_good=False, save_bad=False)
                bad_captions.append(predicted_caption)
                _, _, _, _, orginial_captions, predicted_caption = self.__generate_captions(img_id, deterministic=True, save_good=False, save_bad=False)
                bad_captions.append(predicted_caption)
                self.__save_captions(img_id, bad_captions)
            
        plt.hist(bleu1_scores, bins=10)
        plt.title('Bleu-1 Score Histogram')
        plt.xlabel('Bleu-1 Score')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.__experiment_dir, 'bleu1_hist.png'))

        plt.hist(bleu4_scores, bins=10)
        plt.title('Bleu-4 Score Histogram')
        plt.xlabel('Bleu-4 Score')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.__experiment_dir, 'bleu4_hist.png'))

        write_to_file_in_dir(self.__experiment_dir, 'bleu1_scores.txt', np.mean(bleu1_scores))
        write_to_file_in_dir(self.__experiment_dir, 'bleu4_scores.txt', np.mean(bleu4_scores))

    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))

    def __save_captions(self, img_id, predicted_captions):
        with open(os.path.join('./generated_captions', str(img_id) + '_experiment_caption.txt'), 'w') as f:
                f.write("Img_Id:\n")
                f.write(str(img_id))
                if img_id in self.__goodids:
                    f.write("\nPredicted Caption (Good Image):\n")
                else:
                    f.write("\nPredicted Caption (Bad Image):\n")
                for caption in predicted_captions:
                    f.write("\nCaption:\n")
                    f.write(caption + "\n")
        