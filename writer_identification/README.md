# Purpose

The purpose of this folder is to train and use a model to recognize writers given lines of written text.
The initial model was trained on a subset of writers from the IAM offline dataset https://fki.tic.heia-fr.ch/databases/iam-handwriting-database.

Lines can be cleaned before training using fix-lines.py.

```
python fix-lines.py  --directory <iam lines directory> --destination <fixed lines directory>
```

Training requires a training CSV file that provides a label for each line of text image.

# Training

See training_writer_id.ipynb


# Evalauting

See evaluation.ipynb