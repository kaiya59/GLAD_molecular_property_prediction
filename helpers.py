from tqdm import trange
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
import torch
from torch_geometric.data import Data, DataLoader

electronegativity = {
    'H': 2.2,
    'LI': 0.98,
    'BE': 1.57,
    'B': 2.04,
    'C': 2.55,
    'N': 3.04,
    'O': 3.44,
    'F': 3.98,
    'NA': 0.93,
    'MG': 1.31,
    'AL': 1.61,
    'SI': 1.9,
    'P': 2.19,
    'S': 2.58,
    'CL': 3.16,
    'K': 0.82,
    'CA': 1.0,
    'SC': 1.36,
    'TI': 1.54,
    'V': 1.63,
    'CR': 1.66,
    'MN': 1.55,
    'FE': 1.83,
    'CO': 1.88,
    'NI': 1.91,
    'CU': 1.9,
    'ZN': 1.65,
    'GA': 1.81,
    'GE': 2.01,
    'AS': 2.18,
    'SE': 2.55,
    'BR': 2.96,
    'RB': 0.82,
    'SR': 0.95,
    'Y': 1.22,
    'ZR': 1.33,
    'NB': 1.6,
    'MO': 2.16,
    'TC': 1.9,
    'RU': 2.2,
    'RH': 2.28,
    'PD': 2.2,
    'AG': 1.93,
    'CD': 1.69,
    'IN': 1.78,
    'SN': 1.96,
    'SB': 2.05,
    'TE': 2.1,
    'I': 2.66,
    'CS': 0.79,
    'BA': 0.89,
    'LA': 1.1,
    'CE': 1.12,
    'PR': 1.13,
    'ND': 1.14,
    'PM': 1.13,
    'SM': 1.17,
    'EU': 1.2,
    'GD': 1.2,
    'TB': 1.1,
    'DY': 1.22,
    'HO': 1.23,
    'ER': 1.24,
    'TM': 1.25,
    'YB': 1.1,
    'LU': 1.27,
    'HF': 1.3,
    'TA': 1.5,
    'W': 2.36,
    'RE': 1.9,
    'OS': 2.2,
    'IR': 2.2,
    'PT': 2.28,
    'AU': 2.54,
    'HG': 2.0,
    'TL': 1.62,
    'PB': 2.33,
    'BI': 2.02,
    'PO': 2.0,
    'AT': 2.2,
    'FR': 0.7,
    'RA': 0.9,
    'AC': 1.1,
    'TH': 1.3,
    'PA': 1.5,
    'U': 1.38,
    'NP': 1.36,
    'PU': 1.28,
    'AM': 1.3,
    'CM': 1.3,
    'BK': 1.3,
    'CF': 1.3,
    'ES': 1.3,
    'FM': 1.3,
    'MD': 1.3,
    'NO': 1.3,
    'LR': 1.3
}

def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return mol, scaffold

def has_ring(smiles):
    mol = Chem.MolFromSmiles(smiles)
    AllChem.GetSymmSSSR(mol)
    return Chem.Mol.GetRingInfo(mol).NumRings() > 0


def deconstruct_mol(smiles):

    ###  Molecule = ring fragments + side-chain fragments
    mol = Chem.MolFromSmiles(smiles)
    AllChem.GetSymmSSSR(mol)
    rings = mol.GetRingInfo().AtomRings()
    ring_atoms = [i for ring in rings for i in ring]
    bonds = mol.GetBonds()
    bonds = [bond for bond in bonds if bond.GetBondType() == Chem.BondType.SINGLE or bond.GetBondType() == Chem.BondType.DOUBLE]

    splitting_bonds, sub_molecule_list = [], []
    for bond in bonds:
        begin_atom = bond.GetBeginAtom().GetIdx()
        end_atom = bond.GetEndAtom().GetIdx()
        flag = False
        if begin_atom in ring_atoms and end_atom in ring_atoms:     # if both begin_atom and end_atom are ring atoms:
            rings_contain_begin_atom = set([i for i, ring in enumerate(rings) if begin_atom in ring])
            rings_contain_end_atom = set([i for i, ring in enumerate(rings) if end_atom in ring])
            
            if len(rings_contain_begin_atom.intersection(rings_contain_end_atom)) == 0:    # if they belongs to different rings
                if bond.GetBondType() == Chem.BondType.SINGLE:
                    flag = True     # Break molecule

        if bond.GetBeginAtom().GetSymbol() != '*' and bond.GetEndAtom().GetSymbol() != '*':
            if (begin_atom in ring_atoms and end_atom not in ring_atoms) or (begin_atom not in ring_atoms and end_atom in ring_atoms):  # one atom is in ring and the other is not
                if bond.GetBondType() == Chem.BondType.SINGLE:
                    flag = True     # Break molecule
                
        if flag:
            splitting_bonds.append(bond)


    ###  Molecule = (ring fragment + side-chain fragment)s
    # splitting_atoms, splitting_bonds, sub_molecule_list = [], [], []
    # mol, scaffold = get_scaffold(smiles)
    # AllChem.GetSymmSSSR(scaffold)
    # rings = mol.GetRingInfo().AtomRings()
    # for bond in scaffold.GetBonds():
    #     if bond.GetBondType() == Chem.BondType.SINGLE:
    #         begin_atom, end_atom = bond.GetBeginAtom(), bond.GetEndAtom()
            
    #         if (begin_atom.IsInRing() and not end_atom.IsInRing()) or (not begin_atom.IsInRing() and end_atom.IsInRing()):
    #             splitting_atoms.append([begin_atom.GetAtomMapNum(), end_atom.GetAtomMapNum()])

    #         if (begin_atom.IsInRing() and end_atom.IsInRing()):
    #             flag = True
    #             for ring in rings:
    #                 if begin_atom.GetAtomMapNum() in ring and end_atom.GetAtomMapNum() in ring: # if begin and end atoms belongs to a sane ring
    #                     flag = False # don't split
    #                     break
    #             if flag:
    #                 splitting_atoms.append([begin_atom.GetAtomMapNum(), end_atom.GetAtomMapNum()])

    # for _ in splitting_atoms:
    #     bond = mol.GetBondBetweenAtoms(_[0], _[1])
    #     splitting_bonds.append(bond)
        
    if splitting_bonds != []:
        fragments = Chem.FragmentOnBonds(mol, [bond.GetIdx() for bond in splitting_bonds], addDummies=True)
        
        #######

        frags = Chem.GetMolFrags(fragments)
        for frag in frags:
            sub_molecule = Chem.RWMol()
            cnt = 0
            map_idx = {}
            dummy_idx = []
            for idx in frag:
                if idx < len(mol.GetAtoms()):
                    atom = mol.GetAtomWithIdx(idx)
                    sub_molecule.AddAtom(Chem.Atom(atom.GetAtomicNum()))
                else:
                    sub_molecule.AddAtom(Chem.Atom(0))
                    dummy_idx.append(cnt)
                map_idx[idx] = cnt # map_idx{<idx in original molecule>} : <idx in fragment>
                cnt += 1
                    
            for bond in mol.GetBonds():
                if bond.GetBeginAtomIdx() in frag and bond.GetEndAtomIdx() in frag:
                    sub_molecule.AddBond(map_idx[bond.GetBeginAtomIdx()], map_idx[bond.GetEndAtomIdx()], bond.GetBondType())

                    Begin_atom = sub_molecule.GetAtomWithIdx(map_idx[bond.GetBeginAtomIdx()])
                    Begin_atom.SetAtomMapNum(bond.GetBeginAtomIdx())
                    End_atom = sub_molecule.GetAtomWithIdx(map_idx[bond.GetEndAtomIdx()])
                    End_atom.SetAtomMapNum(bond.GetEndAtomIdx())

                if bond.GetBeginAtomIdx() in frag and bond.GetEndAtomIdx() not in frag:
                    p = dummy_idx.pop(0)
                    sub_molecule.AddBond(map_idx[bond.GetBeginAtomIdx()], p, bond.GetBondType())

                    Begin_atom = sub_molecule.GetAtomWithIdx(map_idx[bond.GetBeginAtomIdx()])
                    Begin_atom.SetAtomMapNum(bond.GetBeginAtomIdx())
                    End_atom = sub_molecule.GetAtomWithIdx(p)
                    End_atom.SetAtomMapNum(bond.GetEndAtomIdx())

                if bond.GetBeginAtomIdx() not in frag and bond.GetEndAtomIdx() in frag:
                    p = dummy_idx.pop(0)
                    sub_molecule.AddBond(p, map_idx[bond.GetEndAtomIdx()], bond.GetBondType())

                    Begin_atom = sub_molecule.GetAtomWithIdx(p)
                    Begin_atom.SetAtomMapNum(bond.GetBeginAtomIdx())
                    End_atom = sub_molecule.GetAtomWithIdx(map_idx[bond.GetEndAtomIdx()])
                    End_atom.SetAtomMapNum(bond.GetEndAtomIdx())

            sub_molecule = sub_molecule.GetMol()
            smiles = [i for i in Chem.MolToSmiles(sub_molecule).split('.') if len(i) > 1][0]
            sub_molecule = Chem.MolFromSmiles(smiles)
            sub_molecule_list.append(sub_molecule)

    else:
        sub_molecule_list.append(mol)

    return sub_molecule_list

def reconstruct_mol(sub_molecule_list):
    frag_dict = {}
    for i, frag1 in enumerate(sub_molecule_list):
        bonds = []
        bond_types = []
        for atom in frag1.GetAtoms():
            if atom.GetSymbol() == '*':
                map_idx = atom.GetAtomMapNum()
                bond = atom.GetBonds()[0].GetBondType()
                for j, frag2 in enumerate(sub_molecule_list):
                    if frag2 != frag1:
                        for atom in frag2.GetAtoms():
                            idx = atom.GetAtomMapNum()
                            if idx == map_idx:
                                bonds.append(j)
                                bond_types.append(bond)
        frag_dict[i] = {'frag': frag1, 'bonds': bonds, 'bond_types': bond_types}
    return frag_dict


def mol_to_pyg_data(mol, bonds, bond_types):

    # Get atom features (atomic number)
    atom_features = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol.upper() in electronegativity.keys():
            en = electronegativity[symbol.upper()]
        else:
            en = 0
        # en = atom.GetAtomicNum()
        atom_features.append(en)
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
    
    return Data(node_attr=node_attr, edge_index=edge_index, edge_attr=edge_attr, bonds=bonds, bond_types=bond_types)


def create_dataloader(df, shuffle=True):
    graph_data_list = []

    for i in trange(len(df), desc='Creating data loader'):
        donor = df.loc[i, 'SD']
        acceptor = df.loc[i, 'SA']
        PCE = df.loc[i, 'PCE']
        donor_dict = reconstruct_mol(deconstruct_mol(donor))
        acceptor_dict = reconstruct_mol(deconstruct_mol(acceptor))

        donor_fragments, donor_bonds, donor_bond_types = [], [], []
        for element in donor_dict:
            donor_fragments.append(donor_dict[element]['frag'])
            donor_bonds.append(donor_dict[element]['bonds'])
            donor_bond_types.append(donor_dict[element]['bond_types'])

        acceptor_fragments, acceptor_bonds, acceptor_bond_types = [], [], []
        for element in acceptor_dict:
            acceptor_fragments.append(acceptor_dict[element]['frag'])
            acceptor_bonds.append(acceptor_dict[element]['bonds'])
            acceptor_bond_types.append(acceptor_dict[element]['bond_types'])

        graph_donor = [mol_to_pyg_data(mol, bonds, bond_types) for mol, bonds, bond_types in zip(donor_fragments, donor_bonds, donor_bond_types)]
        graph_acceptor = [mol_to_pyg_data(mol, bonds, bond_types) for mol, bonds, bond_types in zip(acceptor_fragments, acceptor_bonds, acceptor_bond_types)]
        graph_data_point = [graph_donor, graph_acceptor]
        graph_data_list.append([graph_data_point, PCE])

    loader = DataLoader(graph_data_list, batch_size=1, shuffle=shuffle)

    return loader