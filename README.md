# Graph-Based Diffusion Model for Protein Conformations
A project for CS5284

**Group Members**
- Nico Arsenio Liman
- Jerald Han
- Goh Siow Chuen

## Dataset
- Main Dataset can be downloaded from https://zenodo.org/records/8388270
- Clean protein files IDs are listed in `data/clean_pdb_id.txt`

## Sequence of Notebooks
### EGNN Diffusion (./eGNN_n_DDPM/)
#### Notebooks
- `eGNN.ipynb` - The main notebook where the main model is trained and several attempts at debuggings were made
- `data_processing.ipynb` - Visualization of the dataset as well as testing of the DDPM object to ensure that the proteins were noised correctly.
#### Python Scripts
Initially we looked at modifying GVP model (https://openreview.net/forum?id=1YLJDvSx6J4) to our use case. Nevertheless, this was dropped as generating structures from their input embeddings is non-trivial and would take too much time. 
- `data_read.py` Several functions to extract atom coordinates, constrcut kNN graphs of proteins, radial basis function embeddings as proposed by the GVP
#### Misc
- `train_list.csv` The PDB IDs that the model was trained on
- `val_list.csv` PDB IDs that were not trained on, we use some of the sequences here to generate the protein structures
- `./generated_pdb/` Some of the predicted protein structures, we recommend using ChimeraX/PyMol for visualization
####  Unused
- `igso3.py` We initially considered representing the directionality of the amino acid's side chain as a unit vector, consequently a different kind of diffusion is necessary. i.e., we need to diffuse the direction of said unit vector while maintaining it's magnitude. igso3 diffusion diffuses the angle of the unit vector to a uniform distribution. (See https://openreview.net/pdf?id=oDRQGo8I7P)
-  `modified_diffusion.py` Wrapper object combining conventional DDPM diffusion and igso3 diffusion
### Equivariant Diffusion
- `2_1_edm_testing.ipynb` - Equivariant Diffusion Model

### Latent Diffusion 
- `3_1_protein_latent_diffusion.ipynb` - the main Latent Diffusion for Protein Notebook
- `3_2_qm9_latent_diffusion.ipynb` - side experiments building Latent Diffusion model for simpler molecules with QM9 Datasets
