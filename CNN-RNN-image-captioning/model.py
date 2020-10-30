import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
        
        
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        
        super(DecoderRNN, self).__init__()
         
        # set class variables
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.drop_out = 0.1
        
        # define model layers
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers, batch_first=True, dropout=self.drop_out)
        self.drop = nn.Dropout(0.25)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
        
    def forward(self, features, captions):
        
        # ignore the last output
        captions = captions[:, :-1]
        embeddings = self.embedding(captions)
        
        # concatinate features to embeddings
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)        
        out, hidden = self.lstm(inputs)
        out = self.fc(out)
        
        return out
        
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        # to store the predicted final set of words
        result = []
        
        for i in range(max_len):
            # pas through lstm layer to get output and state
            out_lstm, states = self.lstm(inputs, states)
            out_lstm = out_lstm.squeeze(1)
            out_lstm = out_lstm.squeeze(1)
            
            # pass through linear layer to get probability distribution over vocabs
            out_fc = self.fc(out_lstm)
                     
            # Find the index of max output
            pred = out_fc.max(1)[1]
                       
            # store the indeces of max out to result
            result.append(pred.item())
            
            # prepare the pred to be able to pass to LASTM layer again (repeat until we hit the max_len).
            inputs = self.embedding(pred).unsqueeze(1)
                        
        return result
    