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

# import wandb
# wandb.login()

from helpers import electronegativity, deconstruct_mol, reconstruct_mol

random_seed = 91
torch.manual_seed(random_seed)

################################# Hyperparameters #################################
lr = 0.0005
w0, w1 = 1.0, 1.0
epochs = 100
model_type = 'attentivefp'
num_heads = 1
batch_size = 1
hidden_dim = 128
best_checkpoint = f'best_{model_type}.pth'
last_checkpoint = f'last_{model_type}.pth'
checkpoint_path = 'loss.json'
train_data_path = './CSV/SIDER_train.csv'
val_data_path = './CSV/SIDER_val.csv'
test_data_path = './CSV/SIDER_test.csv'

if torch.cuda.is_available():
    print("[INFO] Current CUDA device:", torch.cuda.current_device())
else:
    print("[INFO] CUDA is not available. Using CPU.")

# run = wandb.init(
#     project="OPV-attentivefp-hopv15",
#     config={
#         "learning_rate": lr,
#         "epochs": epochs,
#     },
# )

################################# Helper functions #################################

class AttentiveFPModel(torch.nn.Module):
    def __init__(self, input_dim=1, hidden_dim=hidden_dim, edge_dim=1, num_layers=6, num_timesteps=1, output_dim=1):
        super(AttentiveFPModel, self).__init__()
        self.layer1 = AttentiveFP(input_dim, hidden_dim, output_dim, edge_dim, num_layers=num_layers, num_timesteps=num_timesteps)

    def forward(self, data):
        node_attr, edge_index, edge_attr = data.node_attr, data.edge_index, data.edge_attr
        if edge_index.size()[0] == 0: # no bonds between 2 atoms
            edge_index = torch.tensor([[0,1], [1,0]])
            edge_attr = torch.tensor([0.0, 0.0]).view(-1, 1)
            node_attr = torch.cat([node_attr, torch.tensor([[0]])]).view(-1, 1)
        batch = torch.zeros(node_attr.size(0), dtype=torch.long)
        node_attr = self.layer1(node_attr, edge_index, edge_attr, batch)
        return node_attr


def mol_to_pyg_data(mol):

    atom_features = []

    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol.upper() in electronegativity.keys():
            en = electronegativity[symbol.upper()]
        else:
            en = 0.001
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
    
    return Data(node_attr=node_attr, edge_index=edge_index, edge_attr=edge_attr)#, mass=mass)


def create_dataloader(df, shuffle=True):
    graph_data_list = []

    for i in trange(len(df), desc='Creating data loader'):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(df.loc[i, 'smiles']))
        mol =Chem.MolFromSmiles(smiles)

        label = df.loc[i, 'Hepatobiliary disorders'].astype(float)
        # label = torch.tensor([df.loc[i, 'CT_TOX'], df.loc[i, 'FDA_APPROVED']], dtype=torch.float32)

        graphs_data = mol_to_pyg_data(mol)
        graph_data_list.append([graphs_data, label])

    loader = DataLoader(graph_data_list, batch_size=batch_size, shuffle=shuffle)

    return loader
    

################################# Main function #################################

if __name__ == '__main__':
    
    ###### Create dataloaders ######
    df_train = pd.read_csv(train_data_path) #.iloc[:2]
    df_val = pd.read_csv(val_data_path) #.iloc[:2]
    df_test = pd.read_csv(test_data_path) #.iloc[:2]

    train_loader = create_dataloader(df_train, shuffle=True)
    val_loader = create_dataloader(df_val, shuffle=True)
    test_loader = create_dataloader(df_test, shuffle=True)

    # print('[INFO] Load training data')
    # with open('./PKL/HIV_train.pkl', 'rb') as f:
    #     train_data = pickle.load(f)
    # print('[INFO] Load validation data')
    # with open('./PKL/HIV_val.pkl', 'rb') as f:
    #     val_data = pickle.load(f)
    # print('[INFO] Load test data')
    # with open('./PKL/HIV_test.pkl', 'rb') as f:
    #     test_data = pickle.load(f)

    # train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    # val_loader = DataLoader(val_data, batch_size=1, shuffle=True)
    # test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    ###### MODELING ######
    model = AttentiveFPModel()
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
            graph_data = data[0]
            predicted_score = model(graph_data)
            logit = torch.sigmoid(predicted_score)
            loss = criterion(logit.squeeze(), label.squeeze())
            # if label.squeeze() == 0:
            #     loss = criterion(logit.squeeze(), label.squeeze())*w0
            # else:
            #     loss = criterion(logit.squeeze(), label.squeeze())*w1

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
                graph_data = data[0]
                predicted_score = model(graph_data)
                logit = torch.sigmoid(predicted_score)
                loss = criterion(logit.squeeze(), label.squeeze())
                LABEL1.append(label.squeeze())
                LOGIT1.append(logit.squeeze())
                # LABEL2.append(label.squeeze()[1])
                # LOGIT2.append(logit.squeeze()[1])

                val_loss += loss.item()
                
        # print('Logit value: ', np.array(LOGIT).min(), np.array(LOGIT).max())
        auc = roc_auc_score(LABEL1, LOGIT1)
        # auc2 = roc_auc_score(LABEL2, LOGIT2)
        # auc = (auc1 + auc2)/2
        print('AUC: ', auc )#, auc1, auc2, auc)
        if auc >= best_score:
            torch.save(model.state_dict(), best_checkpoint)
            print(f'[INFO] Save best checkpoint with auc = {auc}')
            best_score = auc
            t_loss = train_loss

        # Step the learning rate scheduler
        current_lr = optimizer.param_groups[0]['lr']
        # wandb.log({"learning_rate": current_lr}, step=epoch)
        scheduler.step()

        ###### Print loss and save checkpoints ######
        checkpoint_loss[epoch] = {'train loss': train_loss/len(df_train), 'val loss': val_loss/len(df_val)}
        # wandb.log({'train loss': train_loss/len(df_train), 'val loss': val_loss/len(df_val)})
        # print(f'Train loss: {train_loss/len(df_train)}, Val loss: {val_loss/len(df_val)}')
            

    ###### Save last checkpoint ######
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_loss, f)
    torch.save(model.state_dict(), last_checkpoint)
    # wandb.save("model.h5")

    print('Best train loss: ', t_loss/len(df_train))
    print('Best val loss: ', best_score/len(df_val))

    ############ Load best model for testing ############
    print('[INFO] Testing ...')

    model = AttentiveFPModel()

    model.load_state_dict(torch.load(best_checkpoint))
    test_loss = 0
    model.eval()
    LABEL1, LOGIT1, LABEL2, LOGIT2 = [], [], [], []
    with torch.no_grad():
        for data in tqdm(test_loader):
            label = data[1]
            graph_data = data[0]
            predicted_score = model(graph_data)
            logit = torch.sigmoid(predicted_score)
            loss = criterion(logit.squeeze(), label.squeeze())
            LABEL1.append(label.squeeze())
            LOGIT1.append(logit.squeeze())
            # LABEL2.append(label.squeeze()[1])
            # LOGIT2.append(logit.squeeze()[1])
            
        # print(np.array(LOGIT).min(), np.array(LOGIT).max())
        auc1 = roc_auc_score(LABEL1, LOGIT1)
        # auc2 = roc_auc_score(LABEL2, LOGIT2)
        # auc = (auc1 + auc2)/2

        print(f'Test auc: {auc}')