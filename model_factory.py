################################################################################
# CSE 151B: Programming Assignment 3
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin, Yash, James
# Fall 2022
################################################################################
from cProfile import label
import torch
from torchvision.models import resnet50
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda")
torch.autograd.set_detect_anomaly(True)

class CustomCNN(nn.Module):
    '''
    A Custom CNN (Task 1) implemented using PyTorch modules based on the architecture in the PA writeup. 
    This will serve as the encoder for our Image Captioning problem.
    '''
    def __init__(self, outputs):
        '''
        Define the layers (convolutional, batchnorm, maxpool, fully connected, etc.)
        with the correct arguments
        
        Parameters:
            outputs => the number of output classes that the final fully connected layer
                       should map its input to
        '''
        super(CustomCNN, self).__init__()
        # TODO

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.ann = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 1024),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(1024, outputs)
        )

    def forward(self, x):
        '''
        Pass the input through each layer defined in the __init__() function
        in order.

        Parameters:
            x => Input to the CNN
        '''
        # TODO
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.ann(x)
        x = self.fc(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

class CNN_LSTM(nn.Module):
    '''
    An encoder decoder architecture.
    Contains a reference to the CNN encoder based on model_type config value.
    Contains an LSTM implemented using PyTorch modules. This will serve as the decoder for our Image Captioning problem.
    '''
    def __init__(self, config_data, vocab):
        '''
        Initialize the embedding layer, LSTM, and anything else you might need.
        '''
        super(CNN_LSTM, self).__init__()
        self.vocab = vocab
        self.hidden_size = config_data['model']['hidden_size']
        self.embedding_size = config_data['model']['embedding_size']
        self.model_type = config_data['model']['model_type']
        self.max_length = config_data['generation']['max_length']
        self.deterministic = config_data['generation']['deterministic']
        self.temp = config_data['generation']['temperature']

        # TODO

        self.encoder = CustomCNN(self.embedding_size) if self.model_type == 'Custom' else resnet50(pretrained=True)
        if self.model_type != 'Custom':
            self.encoder.fc = nn.Linear(2048, self.embedding_size)
        self.encoder = self.encoder.to(device)

        self.translator = nn.Linear(self.hidden_size, len(self.vocab.word2idx))
        self.translator = self.translator.to(device)

        self.embedding = nn.Embedding(len(self.vocab.word2idx), self.embedding_size)
        self.embedding = self.embedding.to(device)

        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, num_layers=2, batch_first=True)
        self.lstm = self.lstm.to(device)

    def forward(self, images, captions, learning=True, teacher_forcing=False, deterministic=False, criterion=None, optimizer=None):
        '''
        Forward function for this model.
        If teacher forcing is true:
            - Pass encoded images to the LSTM at first time step.
            - Pass each encoded caption one word at a time to the LSTM at every time step after the first one.
        Else:
            - Pass encoded images to the LSTM at first time step.
            - Pass output from previous time step through the LSTM at subsequent time steps
            - Generate predicted caption from the output based on whether we are generating them deterministically or not.
        '''
        # TODO

        batch_size = images.shape[0]
        h = torch.zeros(2, batch_size, self.hidden_size).to(device)
        c = torch.zeros(2, batch_size, self.hidden_size).to(device)
        hidden = (h, c)

        if criterion is not None and optimizer is not None:
            criterion = criterion.to(device)

        images = images.to(device)
        encoded = self.encoder(images)

        pred = torch.zeros((captions.shape[0], captions.shape[1])).to(device)

        if teacher_forcing:
            sequence = self.embedding(captions).to(device)
            sequence = torch.cat((encoded.unsqueeze(1), sequence), dim=1)
            output, hidden = self.lstm(sequence, hidden)
            output = self.translator(output)
            output = output[:, :-1, :]

            # cross entropy loss on sequence
            loss = criterion(output.reshape(-1, len(self.vocab.word2idx)), captions.reshape(-1))
            
            if learning:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        else:
            encoded = encoded.unsqueeze(1)
            output, hidden = self.lstm(encoded, hidden)
            output = output.squeeze(1)
            output = self.translator(output)

            word = None
            loss = None

            if deterministic:
                word = torch.argmax(output, dim=1).unsqueeze(1)
            else:
                word = torch.multinomial(F.softmax(output/self.temp, dim=1), 1)

            pred[:, 0] = word.squeeze(1)

            for i in range(1, self.max_length):
                output = self.embedding(word).to(device)
                output, hidden = self.lstm(output, hidden)
                output = self.translator(output)
                output = output.squeeze(1)

                if deterministic:
                    word = torch.argmax(output, dim=1).unsqueeze(1)
                else:
                    word = torch.multinomial(F.softmax(output/self.temp, dim=1), 1)

                pred[:, i] = word.squeeze(1)
        
        return pred, loss


        """
        encoded = encoded.unsqueeze(1)
        output, hidden = self.lstm(encoded, hidden)
        output = output.squeeze(1)
        output = self.translator(output)
            
        loss=None
        best=None

        if not generation:
            l = criterion(output, captions[:, 0])
            loss=[]
            loss.append(l.item())

            if teacher_forcing and learning:
                optimizer.zero_grad()
                l.backward(retain_graph=True)

        if deterministic:
            best = F.softmax(output, dim=1)
            best = torch.argmax(best, dim=1).unsqueeze(1)
        else:
            best = F.softmax(output/self.temp, dim=1)
            best = torch.multinomial(best, 1)
        
        predicted_caption = torch.zeros((captions.shape[0], captions.shape[1])).to(device)
        best = best.squeeze(1)
        predicted_caption[:, 0] = best

        for i in range(1, captions.size(1)):
            if teacher_forcing:
                output = self.embedding(captions[:, i-1])
            else:
                output = self.embedding(best)

            output = output.unsqueeze(1)
            output, hidden = self.lstm(output, hidden)
            output = output.squeeze(1)
            output = self.translator(output)

            if not generation:
                l = criterion(output, captions[:, i])
                loss.append(l.item())

                if teacher_forcing and learning:
                    l.backward(retain_graph=True)
            
            if deterministic:
                best = F.softmax(output, dim=1)
                best = torch.argmax(best, dim=1)
                best = best.unsqueeze(1)
            else:
                best = F.softmax(output/self.temp, dim=1)
                best = torch.multinomial(best, 1)
            
            best = best.squeeze(1)
            predicted_caption[:, i] = best

        if not generation:
            loss = np.mean(loss)

            if learning:
                optimizer.step()

        return predicted_caption, loss
        """

def get_model(config_data, vocab):
    '''
    Return the LSTM model
    '''
    return CNN_LSTM(config_data, vocab)
