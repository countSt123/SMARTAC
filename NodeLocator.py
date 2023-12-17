# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn import Parameter
class NodeLocator(nn.Module):
    def __init__(self, vocab_size, label_size, query_length, embedding_size, lstm_size, dropout):
        super(NodeLocator, self).__init__()
        self.lstm_size = lstm_size
        self.vocab_size = vocab_size
        self.label_size = label_size
        self.query_length = query_length
        self.embedding_size = embedding_size
        self.lstm_size = lstm_size
        self.query_embedding = nn.Embedding(self.vocab_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.word_encoder = nn.LSTM(input_size = embedding_size, hidden_size=lstm_size, num_layers=1, batch_first=True)
        self.word_encoder_2 = nn.LSTM(input_size = embedding_size, hidden_size=lstm_size, num_layers=1, batch_first=True)

        self.attention_fc_1 = nn.Linear(lstm_size , lstm_size)
        self.attention_fc_2 = nn.Linear(lstm_size, lstm_size)

        
        
        self.class_fc_1_1 = nn.Linear(2*lstm_size, self.label_size)
        self.class_fc_1_2 = nn.Linear(lstm_size, 2*self.lstm_size)
        
    def forward(self, query_data, object_data):
        object_data = object_data.long()
        query_data = query_data.long()
        shape_2 =query_data.shape[1]
        shape_3 =query_data.shape[2]

        query_input = self.query_embedding(query_data)
        object_input = self.query_embedding(object_data)
        query_input = query_input.view(query_input.shape[0], query_input.shape[1]*query_input.shape[2],query_input.shape[3])
      
        query_input = self.dropout(query_input)
        
        query_encoder_outputs, _ = self.word_encoder(query_input)

        query_encoder_outputs = self.dropout(query_encoder_outputs)
        query_encoder_outputs = self.attention_fc_1(query_encoder_outputs)
        

        _, (object_encoder_outputs, _)  = self.word_encoder_2(object_input)
        object_encoder_outputs = self.attention_fc_2(object_encoder_outputs)

        object_encoder_outputs = object_encoder_outputs.permute(1, 2, 0)

        attention_results = torch.softmax(torch.matmul(query_encoder_outputs, object_encoder_outputs), dim=1)
        attention_results= query_encoder_outputs * attention_results

        attention_results = attention_results.reshape(attention_results.shape[0], shape_2, shape_3, attention_results.shape[2])
        attention_results = torch.sum(attention_results, dim=2)
        attention_results = self.dropout(attention_results)
        preds_1 = self.class_fc_1_1(self.class_fc_1_2(attention_results))

        return preds_1








