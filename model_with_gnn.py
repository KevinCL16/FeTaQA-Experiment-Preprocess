import os
import torch
from transformers import T5ForConditionalGeneration
from torch_geometric.nn import GCNConv


class GraphSeq2Seq(torch.nn.Module):
    def __init__(self, input_feature_dim, hidden_channels):
        super().__init__()
        # TODO
        self.conv1 = GCNConv(input_feature_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, 1024)
        self.conv4 = GCNConv(1024, 1024)
        self.conv5 = GCNConv(1024, 512)
        self.softmax = torch.nn.Softmax(dim=1)
        self.decoder = T5ForConditionalGeneration.from_pretrained("t5-small")

    def forward(self, x, edge_index, label, batch, batch_size, train_or_not=True):
        # TODO
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = self.conv4(x, edge_index)
        x = self.conv5(x, edge_index)
        # x = torch.unsqueeze(x, dim=0)
        # x = (x,)

        batch = batch.unsqueeze(dim=1)
        batch_x = torch.hstack((batch, x))
        split_graph_x_in_batch = []
        for i in range(0, batch_size):
            x_per_graph = batch_x[batch_x[:, 0] == i, :]
            split_graph_x_in_batch.append(x_per_graph)

        if train_or_not:
            loss = 0
            for a, each_label in zip(split_graph_x_in_batch, label):
                a = a[a[:, 0] is not None, :, 1:]
                output = self.decoder(encoder_outputs=(a,), labels=each_label)
                each_loss = output.loss
                loss = loss + each_loss

            return loss/batch_size

        else:
            list_of_probs = []
            for a, each_label in zip(split_graph_x_in_batch, label):
                a = a[a[:, 0] is not None, :, 1:]
                output = self.decoder(encoder_outputs=(a,), labels=each_label)
                logits = output.logits
                logits = torch.squeeze(logits, dim=0)
                probs = self.softmax(logits)
                list_of_probs.append(probs)

            return list_of_probs
