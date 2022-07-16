The knowledge graph is made up of two files: ontology.csv and ontology_labels.txt (also a CSV file, apologetically).  These files are designed to be ready by the ontology.py 
script.  The ontology.csv defines the (bi-directional) links between writers and attributes using the global ID.  Each writer and attribute is described in the ontology_labels.txt file with their NAME or IAM Data Set Writer ID.

The columns of ontology_labels.txt are: global id, IAM id or attribute name, node type (writer-id or writer-attribute), and ID specific to the type.

For example: '107,022,writer-id,71':
 (1) 107 this is global ID. 
 (2) This node is writer ID 71 
 (3) This writer is associated with IAM writer 022.  
 
Note: The writer-ids are incremental from 0 to align to one-hot prediction vectors.

Writer profiles are also described in writer_profile.csv. 
