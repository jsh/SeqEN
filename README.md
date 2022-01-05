# SeqEN
Mapping Proteins to Latent Space<br>
<br>
### Method:<br>
A sliding window of size w (*w=20*) generates the input for an AdversarialAutoencoder to map the protein sequence to a line in a latent space (*Dn=10*).  An array of one-hot vectors (*w\*D0*) represents a protein sequence. A single-layered FCN transforms one-hot vectors into vectors of size *D1* to learn the proper amino acid embedding suitable for the mapping. An autoencoder maps the resulting array of (*w\*D1*) to the latent space of *Dn* using Conv1D layers followed by an FCN. A decoder outputs the input by ConvTranspose1d layers and transforms the resulting (*w\*D1*) array to (*w\*D0*). Several adversarial components optimize the mapping. (1) A discriminator learns the difference between the prior distribution and improves the generator (encoder). (2) A classifier receives the latent vectors and improves the mapping using labels provided in the training data for protein domains (Pfam domains). (3) A convolutional decoder, from the latent vector, outputs the secondary structure elements (or Phi and Psi angles) to introduce structural features for improved mapping. <br>
The goal is to map proteins into lines in the latent space; the sequences of similar domains map to similar regions of latent space.  
<br>

### Test Case:<br>
ACT domains are found in many proteins as a modular unit, giving a unique functionality to the protein that carries it. ACT domains are involved in small molecule binding. The ACT domains can form homo/heterodimers, and as a result, they can modulate protein interactions in the presence and absence of small molecules. A peculiar structural feature in this family makes them especially hard to classify with standard methods such as MSA. Mapping ACT domains to a latent space could potentially solve this problem.
