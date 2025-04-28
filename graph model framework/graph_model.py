import torch
import torch.nn as nn
import torch.nn.functional as F
# GNN Model
from torch_geometric.nn import SAGEConv

class GraphZSAGEConv_v5_lin(torch.nn.Module):
    def __init__(self, top_count, edge):
        super(GraphZSAGEConv_v5_lin, self).__init__()
        self.conv1 = SAGEConv(top_count + 4, 1800) # TODO Заменить количество входных признаков на top_count + 4 (топы + A T G C)
        self.conv2 = SAGEConv(1800, 1650)
        self.conv3 = SAGEConv(1650, 1500)
        self.conv4 = SAGEConv(1500, 1350)
        self.conv5 = SAGEConv(1350, 1200)
        self.conv6 = SAGEConv(1200, 1050)
        self.conv7 = SAGEConv(1050, 900)
        self.conv8 = SAGEConv(900, 750)
        self.conv9 = SAGEConv(750, 600)
        self.conv10 = SAGEConv(600, 450)
        self.conv11 = SAGEConv(450, 300)
        self.conv12 = SAGEConv(300, 150)
        self.conv13 = SAGEConv(150, 64)

        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, 2)

        self.edge = edge

    def forward(self, x, edge):
        x = self.conv1(x, edge.cuda())
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv2(x, edge.cuda())
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv3(x, edge.cuda())
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv4(x, edge.cuda())
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv5(x, edge.cuda())
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv6(x, edge.cuda())
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv7(x, edge.cuda())
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv8(x, edge.cuda())
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv9(x, edge.cuda())
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv10(x, edge.cuda())
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv11(x, edge.cuda())
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv12(x, edge.cuda())
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv13(x, edge.cuda())
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=-1)