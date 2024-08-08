import pandas as pd
from tqdm import tqdm, trange
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, AttentiveFP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data, DataLoader
import json
import ipdb
import pickle

from helpers import deconstruct_mol, reconstruct_mol, get_scaffold
from electronegativity import electronegativity
torch.manual_seed(91)

# import wandb
# wandb.login()

################################# Hyperparameters #################################
model_type = 'attentive_fp_text'
num_epoch = 300
batch_size = 1
lr = 0.0005 # 0.001
num_heads = 1
hidden_dim = 64
best_checkpoint = 'best_evolution_text_pc.pth'
# last_checkpoint = f'last_{model_type}.pth'
checkpoint_path = 'loss.json'

train_data_path = f'data_prepare/train_data1.csv'
val_data_path = f'data_prepare/test_data1.csv'
test_data_path = f'data_prepare/val_data1.csv'

with open('./block_fragmentation1.pkl', 'rb') as f:
        text_dict = pickle.load(f)


if torch.cuda.is_available():
    print("[INFO] Current CUDA device:", torch.cuda.current_device())
else:
    print("[INFO] CUDA is not available. Using CPU.")

# run = wandb.init(
#     project="OPV-attentivefp-baseline",
#     config={
#         "learning_rate": lr,
#         "epochs": num_epoch,
#     },
# )

################################# Helper functions #################################

class AttentiveFPModel(torch.nn.Module):
    def __init__(self, input_dim=1, hidden_dim=hidden_dim, edge_dim=1, num_layers=1, num_timesteps=1, output_dim=1): #1, 64
        super(AttentiveFPModel, self).__init__()
        self.layer1 = AttentiveFP(input_dim, hidden_dim, hidden_dim, edge_dim, num_layers=1, num_timesteps=1)
        self.attention = nn.MultiheadAttention(768, 1)
        self.layer2 = AttentiveFP(768+hidden_dim, hidden_dim, output_dim, edge_dim, num_layers=1, num_timesteps=1)
        # self.layer3 = AttentiveFP(hidden_dim, hidden_dim, output_dim, edge_dim, num_layers, num_timesteps)

    def forward(self, data):
        node_attr, edge_index, edge_attr = data.node_attr, data.edge_index, data.edge_attr
        batch = torch.zeros(node_attr.size()[0], dtype=torch.long)
        node_attr = self.layer1(node_attr, edge_index, edge_attr, batch)
        return node_attr


def mol_to_pyg_data(mol, bonds, bond_types, text):

    # Get atom features (atomic number)
    atom_features = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol.upper() in electronegativity.keys():
            en0 = electronegativity[symbol.upper()]
        else:
            en0 = 0
        
        # en1 = atom.GetAtomicNum()
        atom_features.append(en0)
    node_attr = torch.tensor(atom_features, dtype=torch.float).view(-1, 1)

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
        donor = df.loc[i, 'SD']
        acceptor = df.loc[i, 'SA']
        PCE = df.loc[i, 'PCE']
        donor_dict = reconstruct_mol(deconstruct_mol(donor))
        acceptor_dict = reconstruct_mol(deconstruct_mol(acceptor))

        donor_fragments, donor_bonds, donor_bond_types, donor_texts = [], [], [], []
        for i in donor_dict:

            frag = donor_dict[i]['frag']
            for atom in frag.GetAtoms():
                atom.SetAtomMapNum(0)

            donor_fragments.append(donor_dict[i]['frag'])
            donor_bonds.append(donor_dict[i]['bonds'])
            donor_bond_types.append(donor_dict[i]['bond_types'])

            donor_smiles = Chem.MolToSmiles(frag)
            try:
                donor_text = torch.tensor(text_dict[donor_smiles], dtype=torch.float32).view(-1, 768)
            except:
                print('y')
                donor_text = torch.zeros((150, 768), dtype=torch.float32).view(-1, 768)
            donor_texts.append(donor_text)

        acceptor_fragments, acceptor_bonds, acceptor_bond_types, acceptor_texts = [], [], [], []
        for i in acceptor_dict:

            frag = acceptor_dict[i]['frag']
            for atom in frag.GetAtoms():
                atom.SetAtomMapNum(0)

            acceptor_fragments.append(acceptor_dict[i]['frag'])
            acceptor_bonds.append(acceptor_dict[i]['bonds'])
            acceptor_bond_types.append(acceptor_dict[i]['bond_types'])

            acceptor_smiles = Chem.MolToSmiles(frag)
            try:
                acceptor_text = torch.tensor(text_dict[acceptor_smiles], dtype=torch.float32).view(-1, 768)
            except:
                print('y')
                donor_text = torch.zeros((150, 768), dtype=torch.float32).view(-1, 768)
            acceptor_texts.append(acceptor_text)

        graph_donor = [mol_to_pyg_data(mol, bonds, bond_types, donor_text) for mol, bonds, bond_types, donor_text in zip(donor_fragments, donor_bonds, donor_bond_types, donor_texts)]
        graph_acceptor = [mol_to_pyg_data(mol, bonds, bond_types, acceptor_text) for mol, bonds, bond_types, acceptor_text in zip(acceptor_fragments, acceptor_bonds, acceptor_bond_types, acceptor_texts)]
        graph_data_point = [graph_donor, graph_acceptor]
        graph_data_list.append([graph_data_point, PCE])

    loader = DataLoader(graph_data_list, batch_size=batch_size, shuffle=shuffle)

    return loader


if __name__ == '__main__':

    ###### Create dataloaders ######
    df_train = pd.read_csv(train_data_path)#.loc[:20]
    df_val = pd.read_csv(val_data_path)#.loc[:20]
    df_test = pd.read_csv(test_data_path)#.loc[:20]

    train_loader = create_dataloader(df_train, text_dict, shuffle=False)
    val_loader = create_dataloader(df_val, text_dict, shuffle=False)
    test_loader = create_dataloader(df_test, text_dict, shuffle=False)

    ###### MODELING ######

    model = AttentiveFPModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=25, eta_min=0.0001)

    # ###### Start training ######
    best_score = float('inf')
    checkpoint_loss = {}
    for epoch in range(num_epoch):
        train_loss = 0
        model.train()
        for data in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epoch}'):
            PCE = data[1]
            graph_data_list = data[0]
            graph_donor = graph_data_list[0]
            graph_acceptor = graph_data_list[1]
            donor_substructure_embeddings = []
            for donor in graph_donor:
                node_representations = model(donor)
                v = torch.hstack([node_representations, torch.zeros((1, 768-hidden_dim))]) # size: (1, 768)
                donor_text_descriptor = donor['text'] # size: (100, 768)
                Q = v.repeat(donor_text_descriptor.size(0), 1)
                attention_weight = model.attention(Q, donor_text_descriptor, donor_text_descriptor)[1][0]
                attention = (attention_weight.repeat(768,1).T * donor_text_descriptor).sum(0)
                donor_input = torch.hstack([attention, node_representations.squeeze()]).view((1, -1))

                donor_substructure_embeddings.append([donor_input, donor.bonds[0], donor.bond_types[0]])

            acceptor_substructure_embeddings = []
            for acceptor in graph_acceptor:
                node_representations = model(acceptor)
                v = torch.hstack([node_representations, torch.zeros((1, 768-hidden_dim))]) # size: (1, 768)
                acceptor_text_descriptor = acceptor['text'] # size: (100, 768)
                Q = v.repeat(acceptor_text_descriptor.size(0), 1)
                attention_weight = model.attention(Q, acceptor_text_descriptor, acceptor_text_descriptor)[1][0]
                attention = (attention_weight.repeat(768,1).T * acceptor_text_descriptor).sum(0)
                acceptor_input = torch.hstack([attention, node_representations.squeeze()]).view((1, -1))

                acceptor_substructure_embeddings.append([acceptor_input, acceptor.bonds[0], acceptor.bond_types[0]])
            
            donor_node_attr, donor_edge_idx, donor_edge_attr = [], [], []
            for i, element in enumerate(donor_substructure_embeddings):
                donor_node_attr.append(element[0].tolist())
                donor_edge_idx.append([[i, j] for j in element[1]])
                donor_edge_attr.append(element[2])
            donor_edge_idx = [l for k in donor_edge_idx for l in k]
            donor_edge_attr = [l for k in donor_edge_attr for l in k]

            acceptor_node_attr, acceptor_edge_idx, acceptor_edge_attr = [], [], []
            for i, element in enumerate(acceptor_substructure_embeddings):
                acceptor_node_attr.append(element[0].tolist())
                acceptor_edge_idx.append([[i, j] for j in element[1]])
                acceptor_edge_attr.append(element[2])
            acceptor_edge_idx = [l for k in acceptor_edge_idx for l in k]
            acceptor_edge_idx = [(i+len(donor_node_attr), j+len(donor_node_attr)) for (i,j) in acceptor_edge_idx]
            acceptor_edge_attr = [l for k in acceptor_edge_attr for l in k]

            node_attr = donor_node_attr + acceptor_node_attr
            node_attr = torch.tensor(node_attr, dtype=torch.float)
            edge_index = donor_edge_idx + acceptor_edge_idx
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = donor_edge_attr + acceptor_edge_attr
            edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1,1)

            batch = torch.zeros(node_attr.squeeze().size(0), dtype=torch.long)
            predicted_score = model.layer2(node_attr.squeeze(), edge_index, edge_attr, batch)

            loss = criterion(predicted_score.squeeze(), PCE.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        ###### Evaluation ######
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for data in tqdm(val_loader, desc=f'Validation'):
                PCE = data[1]
                graph_data_list = data[0]
                graph_donor = graph_data_list[0]
                graph_acceptor = graph_data_list[1]
                
                donor_substructure_embeddings = []
                for donor in graph_donor:
                    node_representations = model(donor)
                    v = torch.hstack([node_representations, torch.zeros((1, 768-hidden_dim))]) # size: (1, 768)
                    donor_text_descriptor = donor['text'] # size: (100, 768)
                    Q = v.repeat(donor_text_descriptor.size(0), 1)
                    attention_weight = model.attention(Q, donor_text_descriptor, donor_text_descriptor)[1][0]
                    attention = (attention_weight.repeat(768,1).T * donor_text_descriptor).sum(0)
                    donor_input = torch.hstack([attention, node_representations.squeeze()]).view((1, -1))

                    donor_substructure_embeddings.append([donor_input, donor.bonds[0], donor.bond_types[0]])

                acceptor_substructure_embeddings = []
                for acceptor in graph_acceptor:
                    node_representations = model(acceptor)
                    v = torch.hstack([node_representations, torch.zeros((1, 768-hidden_dim))]) # size: (1, 768)
                    acceptor_text_descriptor = acceptor['text'] # size: (100, 768)
                    Q = v.repeat(acceptor_text_descriptor.size(0), 1)
                    attention_weight = model.attention(Q, acceptor_text_descriptor, acceptor_text_descriptor)[1][0]
                    attention = (attention_weight.repeat(768,1).T * acceptor_text_descriptor).sum(0)
                    acceptor_input = torch.hstack([attention, node_representations.squeeze()]).view((1, -1))

                    acceptor_substructure_embeddings.append([acceptor_input, acceptor.bonds[0], acceptor.bond_types[0]])
                
                donor_node_attr, donor_edge_idx, donor_edge_attr = [], [], []
                for i, element in enumerate(donor_substructure_embeddings):
                    donor_node_attr.append(element[0].tolist())
                    donor_edge_idx.append([[i, j] for j in element[1]])
                    donor_edge_attr.append(element[2])
                donor_edge_idx = [l for k in donor_edge_idx for l in k]
                donor_edge_attr = [l for k in donor_edge_attr for l in k]

                acceptor_node_attr, acceptor_edge_idx, acceptor_edge_attr = [], [], []
                for i, element in enumerate(acceptor_substructure_embeddings):
                    acceptor_node_attr.append(element[0].tolist())
                    acceptor_edge_idx.append([[i, j] for j in element[1]])
                    acceptor_edge_attr.append(element[2])
                acceptor_edge_idx = [l for k in acceptor_edge_idx for l in k]
                acceptor_edge_idx = [(i+len(donor_node_attr), j+len(donor_node_attr)) for (i,j) in acceptor_edge_idx]
                acceptor_edge_attr = [l for k in acceptor_edge_attr for l in k]

                node_attr = donor_node_attr + acceptor_node_attr
                node_attr = torch.tensor(node_attr, dtype=torch.float)
                edge_index = donor_edge_idx + acceptor_edge_idx
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                edge_attr = donor_edge_attr + acceptor_edge_attr
                edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1,1)

                batch = torch.zeros(node_attr.squeeze().size(0), dtype=torch.long)
                predicted_score = model.layer2(node_attr.squeeze(), edge_index, edge_attr, batch)
                    
                loss = criterion(predicted_score.squeeze(), PCE.squeeze())
                val_loss += loss.item()

            # Step the learning rate scheduler
            current_lr = optimizer.param_groups[0]['lr']
            # wandb.log({"learning_rate": current_lr}, step=epoch)
            scheduler.step()
            

        ###### Print loss and save checkpoints ######
        checkpoint_loss[epoch] = {'train loss': train_loss/len(df_train), 'val loss': val_loss/len(df_val)}
        # wandb.log({'train loss': train_loss/len(df_train), 'val loss': val_loss/len(df_val)})
        print(f'Train loss: {train_loss/len(df_train)}, Val loss: {val_loss/len(df_val)}')
        if val_loss <= best_score:
            torch.save(model.state_dict(), best_checkpoint)
            print(f'[INFO] Save best checkpoint with val loss = {val_loss/len(df_val)}')
            best_score = val_loss
            t_loss = train_loss

    ##### Save last checkpoint ######
    # with open(checkpoint_path, 'w') as f:
    #     json.dump(checkpoint_loss, f)
    # torch.save(model.state_dict(), last_checkpoint)
    # wandb.save("model.h5")

    print('Best train loss: ', t_loss/len(df_train))
    print('Best val loss: ', best_score/len(df_val))

    ###### Load best model for testing ######
    print('[INFO] Testing ...')

    model = AttentiveFPModel()

    model.load_state_dict(torch.load(best_checkpoint))
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            PCE = data[1]
            graph_data_list = data[0]
            graph_donor = graph_data_list[0]
            graph_acceptor = graph_data_list[1]
            
            donor_substructure_embeddings = []
            for donor in graph_donor:
                node_representations = model(donor)
                v = torch.hstack([node_representations, torch.zeros((1, 768-hidden_dim))]) # size: (1, 768)
                donor_text_descriptor = donor['text'] # size: (N, 768)
                Q = v.repeat(donor_text_descriptor.size(0), 1)
                attention_weight = model.attention(Q, donor_text_descriptor, donor_text_descriptor)[1][0]
                attention = (attention_weight.repeat(768,1).T * donor_text_descriptor).sum(0)
                donor_input = torch.hstack([attention, node_representations.squeeze()]).view((1, -1))

                donor_substructure_embeddings.append([donor_input, donor.bonds[0], donor.bond_types[0]])

            acceptor_substructure_embeddings = []   
            for acceptor in graph_acceptor:
                node_representations = model(acceptor)
                v = torch.hstack([node_representations, torch.zeros((1, 768-hidden_dim))]) # size: (1, 768)
                acceptor_text_descriptor = acceptor['text'] # size: (100, 768)
                Q = v.repeat(acceptor_text_descriptor.size(0), 1)
                attention_weight = model.attention(Q, acceptor_text_descriptor, acceptor_text_descriptor)[1][0]
                attention = (attention_weight.repeat(768,1).T * acceptor_text_descriptor).sum(0)
                acceptor_input = torch.hstack([attention, node_representations.squeeze()]).view((1, -1))

                acceptor_substructure_embeddings.append([acceptor_input, acceptor.bonds[0], acceptor.bond_types[0]])
            
            donor_node_attr, donor_edge_idx, donor_edge_attr = [], [], []
            for i, element in enumerate(donor_substructure_embeddings):
                donor_node_attr.append(element[0].tolist())
                donor_edge_idx.append([[i, j] for j in element[1]])
                donor_edge_attr.append(element[2])
            donor_edge_idx = [l for k in donor_edge_idx for l in k]
            donor_edge_attr = [l for k in donor_edge_attr for l in k]

            acceptor_node_attr, acceptor_edge_idx, acceptor_edge_attr = [], [], []
            for i, element in enumerate(acceptor_substructure_embeddings):
                acceptor_node_attr.append(element[0].tolist())
                acceptor_edge_idx.append([[i, j] for j in element[1]])
                acceptor_edge_attr.append(element[2])
            acceptor_edge_idx = [l for k in acceptor_edge_idx for l in k]
            acceptor_edge_idx = [(i+len(donor_node_attr), j+len(donor_node_attr)) for (i,j) in acceptor_edge_idx]
            acceptor_edge_attr = [l for k in acceptor_edge_attr for l in k]

            node_attr = donor_node_attr + acceptor_node_attr
            node_attr = torch.tensor(node_attr, dtype=torch.float)
            edge_index = donor_edge_idx + acceptor_edge_idx
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = donor_edge_attr + acceptor_edge_attr
            edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1,1)

            batch = torch.zeros(node_attr.squeeze().size(0), dtype=torch.long)
            predicted_score = model.layer2(node_attr.squeeze(), edge_index, edge_attr, batch)

            loss = criterion(predicted_score.squeeze(), PCE.squeeze())
            print(f'Groundtruth: {round(float(PCE.squeeze()), 3)}, Predicted: {round(float(predicted_score.squeeze()), 3)}')
            test_loss += loss.item()

        print(f'Test loss: {test_loss/len(df_test)}')