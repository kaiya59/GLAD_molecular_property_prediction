import pandas as pd
from tqdm import tqdm, trange
import pandas as pd
from rdkit import Chem
import torch
import torch.nn as nn
from torch_geometric.nn import AttentiveFP
from torch_geometric.data import Data, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import numpy as np
import json
import pickle
import ipdb

import wandb
wandb.login()

from helpers import electronegativity, deconstruct_mol, reconstruct_mol

random_seed = 99
torch.manual_seed(random_seed)

################################# Hyperparameters #################################
lr = 0.0005
w0, w1 = 3.0, 1.0
epochs = 75
model_type = 'attentivefp'
num_heads = 2
batch_size = 1
hidden_dim = 64
best_checkpoint = './best_baseline.pth'
last_checkpoint = './last_baseline.pth'
checkpoint_path = 'loss.json'
train_data_path = './data_prepare/train_data.csv'
val_data_path = './data_prepare/val_data.csv'
test_data_path = './data_prepare/test_data.csv'
text_embedding_path = './PKL/BBBP_text_embeddings.pkl'

if torch.cuda.is_available():
    print("[INFO] Current CUDA device:", torch.cuda.current_device())
else:
    print("[INFO] CUDA is not available. Using CPU.")

################################# Helper functions #################################

    
class AttentiveFPModel(torch.nn.Module):
    def __init__(self, input_dim=1, hidden_dim=hidden_dim, edge_dim=1, num_layers=1, num_timesteps=1, output_dim=1): #1, 64
        super(AttentiveFPModel, self).__init__()
        self.layer1 = AttentiveFP(input_dim, hidden_dim, hidden_dim, edge_dim, num_layers, num_timesteps)
        self.attention = nn.MultiheadAttention(768, 1)
        self.layer2 = AttentiveFP(hidden_dim+768, hidden_dim, output_dim, edge_dim, num_layers=2, num_timesteps=1)

    def forward(self, data):
        node_attr, edge_index, edge_attr = data.node_attr, data.edge_index, data.edge_attr
        if edge_index.size()[0] == 0:
            edge_index = torch.tensor([[0,1], [1,0]])
            edge_attr = torch.tensor([4.0, 4.0]).view(-1, 1)
            node_attr = torch.cat([node_attr, torch.tensor([[0]])]).view(-1, 1)
        batch = torch.zeros(node_attr.size()[0], dtype=torch.long)
        node_attr = self.layer1(node_attr, edge_index, edge_attr, batch)
        return node_attr



def mol_to_pyg_data(mol, bonds, bond_types, text):

    # Get atom features (atomic number)
    atom_features = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol.upper() in electronegativity.keys():
            en = electronegativity[symbol.upper()]
        else:
            en = 0
        atom_features.append(en)
    
    node_attr = torch.tensor(atom_features, dtype=torch.float).view(-1, 1) #####

    # Get edge indices and edge features (bond types)
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        edge_index.append([start, end])
        edge_index.append([end, start])
        edge_attr.append(bond.GetBondTypeAsDouble())
        edge_attr.append(bond.GetBondTypeAsDouble())
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)
    
    return Data(node_attr=node_attr, edge_index=edge_index, edge_attr=edge_attr, bonds=bonds, bond_types=bond_types, text=text)


def create_dataloader(df, text_dict, shuffle=True):
    graph_data_list = []

    for i in trange(len(df), desc='Creating data loader'):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(df.loc[i, 'smiles']))
        label = df.loc[i, 'FDA_APPROVED'].astype(float)
        bb_dict = reconstruct_mol(deconstruct_mol(smiles))

        fragments, bonds, bond_types, text_ = [], [], [], [] # store all building blocks, bonds, and bond types of a molecule
        for element in bb_dict: # loop over each building block
            m = bb_dict[element]['frag']
            for atom in m.GetAtoms():
                atom.SetAtomMapNum(0)
            smiles = Chem.MolToSmiles(m)
            fragments.append(m)
            bonds.append(bb_dict[element]['bonds'])
            bond_types.append(bb_dict[element]['bond_types'])
            try:
                text = torch.tensor(text_dict[smiles].squeeze(), dtype=torch.float32).view(-1, 768)
            except:
                print(smiles)
                text = torch.zeros(768, dtype=torch.float32).view(-1, 768)
            text_.append(text)

        graphs_data = [mol_to_pyg_data(mol, bond, bond_type, text) for mol, bond, bond_type, text in zip(fragments, bonds, bond_types, text_)] # 
        graph_data_list.append([graphs_data, label])


    loader = DataLoader(graph_data_list, batch_size=batch_size, shuffle=shuffle)

    return loader
    

################################# Main function #################################

if __name__ == '__main__':

    with open(text_embedding_path, 'rb') as f:
        text_dict = pickle.load(f)
    
    ###### Create dataloaders ######
    df_train = pd.read_csv(train_data_path)
    df_val = pd.read_csv(val_data_path)
    df_test = pd.read_csv(test_data_path)

    train_loader = create_dataloader(df_train, text_dict, shuffle=True)
    val_loader = create_dataloader(df_val, text_dict, shuffle=True)
    test_loader = create_dataloader(df_test, text_dict, shuffle=True)

    ###### MODELING ######
    model = AttentiveFPModel() #; model.load_state_dict(torch.load(last_checkpoint))
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=25, eta_min=0.0001)

    ##### Start training ######
    best_score = 0.0
    checkpoint_loss = {}
    for epoch in range(epochs):
        train_loss = 0
        model.train()
        for data in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            label = data[1]
            graph_data_list = data[0] # graph representation of each molecule
            
            bb_embeddings = []
            for bb in graph_data_list: # bb: building block
                bb_em = model(bb)
                v = torch.hstack([bb_em, torch.zeros((1, 768-hidden_dim))]) # size: (1, 768)
                text_descriptor = bb.text # size: (100, 768)
                Q = v.repeat(text_descriptor.size(0), 1)
                attention_weight = model.attention(Q, text_descriptor, text_descriptor)[1][0]
                attention = (attention_weight.repeat(768,1).T * text_descriptor).sum(0)
                em = torch.hstack([attention, bb_em.squeeze()]).view((1, -1))
                bb_embeddings.append([em.squeeze(), bb.bonds[0], bb.bond_types[0]])

            bb_attr, bb_edge_idx, bb_edge_attr = [], [], []
            for i, bb_em in enumerate(bb_embeddings):
                bb_attr.append(bb_em[0].tolist())
                bb_edge_idx.append([[i, j] for j in bb_em[1]])
                bb_edge_attr.append(bb_em[2])
            bb_edge_idx = [l for k in bb_edge_idx for l in k]
            bb_edge_attr = [l for k in bb_edge_attr for l in k]

            node_attr = torch.tensor(bb_attr, dtype=torch.float)
            edge_index = torch.tensor(bb_edge_idx, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(bb_edge_attr, dtype=torch.float).view(-1,1)

            if edge_index.size()[0] == 0 and node_attr.size()[0] == 1:
                edge_index = torch.tensor([[0,1], [1,0]])
                edge_attr = torch.tensor([4.0, 4.0]).view(-1, 1) # edge attr = 4.0 --> no connection
                node_attr = torch.cat([node_attr, torch.zeros((1, hidden_dim+768))])

            if edge_index.size()[0] == 0 and node_attr.size()[0] > 1:
                edge_index = torch.tensor([[0,1], [1,0]])
                edge_attr = torch.tensor([4.0, 4.0]).view(-1, 1)

            batch = torch.zeros(node_attr.squeeze().size(0), dtype=torch.long)
            predicted_score = model.layer2(node_attr.squeeze(), edge_index, edge_attr, batch)
            logit = torch.sigmoid(predicted_score)
            # loss = criterion(logit.squeeze(), label.squeeze())

            if label.squeeze() == 0:
                loss = criterion(logit.squeeze(), label.squeeze())*w0
            else:
                loss = criterion(logit.squeeze(), label.squeeze())*w1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        ###### Evaluation ######
        val_loss = 0
        model.eval()
        LABEL1, LOGIT1, LABEL2, LOGIT2 = [], [], [], []
        with torch.no_grad():
            for data in tqdm(val_loader):
                label = data[1]
                graph_data_list = data[0] # graph representation of each molecule
                
                bb_embeddings = []
                for bb in graph_data_list: # bb: building block
                    bb_em = model(bb)
                    v = torch.hstack([bb_em, torch.zeros((1, 768-hidden_dim))]) # size: (1, 768)
                    text_descriptor = bb.text # size: (100, 768)
                    Q = v.repeat(text_descriptor.size(0), 1)
                    attention_weight = model.attention(Q, text_descriptor, text_descriptor)[1][0]
                    attention = (attention_weight.repeat(768,1).T * text_descriptor).sum(0)
                    em = torch.hstack([attention, bb_em.squeeze()]).view((1, -1))
                    bb_embeddings.append([em.squeeze(), bb.bonds[0], bb.bond_types[0]])
                
                bb_attr, bb_edge_idx, bb_edge_attr = [], [], []
                for i, bb_em in enumerate(bb_embeddings):
                    bb_attr.append(bb_em[0].tolist())
                    bb_edge_idx.append([[i, j] for j in bb_em[1]])
                    bb_edge_attr.append(bb_em[2])
                bb_edge_idx = [l for k in bb_edge_idx for l in k]
                bb_edge_attr = [l for k in bb_edge_attr for l in k]

                node_attr = torch.tensor(bb_attr, dtype=torch.float)
                edge_index = torch.tensor(bb_edge_idx, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(bb_edge_attr, dtype=torch.float).view(-1,1)
                    
                if edge_index.size()[0] == 0 and node_attr.size()[0] == 1:
                    edge_index = torch.tensor([[0,1], [1,0]])
                    edge_attr = torch.tensor([4.0, 4.0]).view(-1, 1)
                    node_attr = torch.cat([node_attr, torch.zeros((1, hidden_dim+768))])

                if edge_index.size()[0] == 0 and node_attr.size()[0] > 1:
                    edge_index = torch.tensor([[0,1], [1,0]])
                    edge_attr = torch.tensor([4.0, 4.0]).view(-1, 1)

                batch = torch.zeros(node_attr.squeeze().size(0), dtype=torch.long)
                predicted_score = model.layer2(node_attr.squeeze(), edge_index, edge_attr, batch)
                logit = torch.sigmoid(predicted_score)
                loss = criterion(logit.squeeze(), label.squeeze())
                LABEL1.append(label.squeeze())
                LOGIT1.append(logit.squeeze())
                val_loss += loss.item()
                
        auc = roc_auc_score(LABEL1, LOGIT1)

        print('AUC: ', auc) # auc1, auc2, auc)
        if auc >= best_score:
            torch.save(model.state_dict(), best_checkpoint)
            print(f'[INFO] Save best checkpoint with auc = {auc}')
            best_score = auc
            t_loss = train_loss

        # Step the learning rate scheduler
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"learning_rate": current_lr}, step=epoch)
        scheduler.step()

        ###### Print loss and save checkpoints ######
        checkpoint_loss[epoch] = {'train loss': train_loss/len(df_train), 'val loss': val_loss/len(df_val)}
        wandb.log({'train loss': train_loss/len(df_train), 'val loss': val_loss/len(df_val), 'val auc': auc})
        print(f'Train loss: {train_loss/len(df_train)}, Val loss: {val_loss/len(df_val)}')
            

    ###### Save last checkpoint ######
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_loss, f)
    torch.save(model.state_dict(), last_checkpoint)
    wandb.save("model.h5")

    print('Best train loss: ', t_loss/len(df_train))
    print('Best val loss: ', best_score/len(df_val))

    ############ Load best model for testing ############
    print('[INFO] Testing ...')

    model = AttentiveFPModel()

    model.load_state_dict(torch.load(best_checkpoint))
    test_loss = 0
    model.eval()
    LABEL, LOGIT = [], []
    with torch.no_grad():
        for data in tqdm(test_loader):
            label = data[1]
            graph_data_list = data[0] # graph representation of each molecule
            
            bb_embeddings = []
            for bb in graph_data_list: # bb: building block
                bb_em = model(bb)
                v = torch.hstack([bb_em, torch.zeros((1, 768-hidden_dim))]) # size: (1, 768)
                text_descriptor = bb.text # size: (100, 768)
                Q = v.repeat(text_descriptor.size(0), 1)
                attention_weight = model.attention(Q, text_descriptor, text_descriptor)[1][0]
                attention = (attention_weight.repeat(768,1).T * text_descriptor).sum(0)
                em = torch.hstack([attention, bb_em.squeeze()]).view((1, -1))
                bb_embeddings.append([em.squeeze(), bb.bonds[0], bb.bond_types[0]])
            
            bb_attr, bb_edge_idx, bb_edge_attr = [], [], []
            for i, bb_em in enumerate(bb_embeddings):
                bb_attr.append(bb_em[0].tolist())
                bb_edge_idx.append([[i, j] for j in bb_em[1]])
                bb_edge_attr.append(bb_em[2])
            bb_edge_idx = [l for k in bb_edge_idx for l in k]
            bb_edge_attr = [l for k in bb_edge_attr for l in k]

            node_attr = torch.tensor(bb_attr, dtype=torch.float)
            edge_index = torch.tensor(bb_edge_idx, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(bb_edge_attr, dtype=torch.float).view(-1,1)

            if edge_index.size()[0] == 0 and node_attr.size()[0] == 1:
                edge_index = torch.tensor([[0,1], [1,0]])
                edge_attr = torch.tensor([4.0, 4.0]).view(-1, 1)
                node_attr = torch.cat([node_attr, torch.zeros((1, hidden_dim + 768))])

            if edge_index.size()[0] == 0 and node_attr.size()[0] > 1:
                edge_index = torch.tensor([[0,1], [1,0]])
                edge_attr = torch.tensor([4.0, 4.0]).view(-1, 1)

            batch = torch.zeros(node_attr.squeeze().size(0), dtype=torch.long)
            predicted_score = model.layer2(node_attr.squeeze(), edge_index, edge_attr, batch)
            logit = torch.sigmoid(predicted_score)
            loss = criterion(logit.squeeze(), label.squeeze())
            LABEL1.append(label.squeeze())
            LOGIT1.append(logit.squeeze())
            
        auc1 = roc_auc_score(LABEL1, LOGIT1)
        print('AUC: ', auc) # auc1, auc2, auc)