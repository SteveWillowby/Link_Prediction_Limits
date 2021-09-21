from ogb.lsc import PCQM4MDataset
from ogb.utils import smiles2graph

# smiles2graph takes a SMILES string as input and returns a graph object
# requires rdkit to be installed.
# You can write your own smiles2graph
graph_obj = smiles2graph('O=C1C=CC(O1)C(c1ccccc1C)O')

# convert each SMILES string into a molecular graph object by calling smiles2graph
# This takes a while (a few hours) for the first run
dataset = PCQM4MDataset(root = "/data/datasets/open_graph_benchmark_LSC_2021/PCQM4M", smiles2graph = smiles2graph)
