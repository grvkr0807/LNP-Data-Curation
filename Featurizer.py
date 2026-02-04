#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors

# -------- One-hot schema (must match CSV file) --------
CATS = {
    'mixing_method': ['handmixed','microfluidics'],
    'model_type':    ['HeLa','A549','Mouse','RAW264.7','HepG2','DC2.4','IGROV1','BeWo_b30','HEK293T','Human_RBC','BMDC','HBEC_ALI','BMDM'],
    'model_target':  ['in_vitro','lung_epithelium','liver','muscle','spleen','multiorgan','lung','heart','kidney'],
    'route_of_administration': ['in_vitro','intravenous','intramuscular','intratracheal'],
    'cargo':         ['mRNA','pDNA','siRNA'],
    'cargo_type':    ['FFL','DNA_barcode','peptide_barcode','hEPO','FVII','GFP'],
}

# Single numeric/boolean metadata columns (unchanged)
META_NUMERIC = [
    "heavy_atoms","rings","aromatic_rings","rotatable_bonds",
    "van_der_waals_molecular_volume","topological_polar_surface_area",
    "hydrogen_bond_donors","hydrogen_bond_acceptors","logp","molar_refractivity",
    "fraction_sp3_carbons","sp3_carbons","nitrogen_count","molecular_weight",
    "has_ester","has_carbonate","has_disulfide",
    "il_molratio","hl_molratio","chl_molratio","peg_molratio",
    "il_to_mrna_massratio"
]

# Build the complete ordered one-hot column list expected in smiles_df
META_OHE = []
for k, levels in CATS.items():
    META_OHE += [f"{k}_{lvl}" for lvl in levels]

# -------------------------------
# 1) RDKit Descriptor Featurizer
# -------------------------------
def RDKit_Descriptors(smiles_df, Nrows):

    X = []
    ipc_index = [42]  # drop Ipc descriptor

    for i in range(Nrows):
        features_full = []

        # --- Descriptors for each constituent in fixed order ---
        for col in ["il_smiles", "hl_smiles", "chl_smiles", "peg_smiles"]:
            smiles = smiles_df.iloc[i][col]

            RDLogger.DisableLog('rdApp.error')
            mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles.strip() else None
            RDLogger.EnableLog('rdApp.error')

            if mol is not None:
                features = np.array(list(Descriptors.CalcMolDescriptors(mol).values()))
                features = np.delete(features, ipc_index)
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                features = np.zeros(209)  # same as before

            features_full.extend(features)

        # --- Append numeric/boolean metadata (single columns) ---
        row = smiles_df.iloc[i]
        for col in META_NUMERIC:
            val = row[col] if col in smiles_df.columns else 0
            if isinstance(val, (bool, np.bool_)):
                val = int(val)
            features_full.append(val)

        # --- Append one-hot metadata (0/1 for each expected column) ---
        for col in META_OHE:
            val = row[col] if col in smiles_df.columns else 0
            # ensure 0/1 numeric
            features_full.append(int(val) if pd.notna(val) else 0)

        X.append(np.array(features_full, dtype=object))

    return np.array(X, dtype=object)
