# Handwriting Recognition with Novelty

Provide training and tests set for Handwriting Recognition with Novelty.

This repository contains the code for the ICDAR 2021 paper, "Handwriting Recognition with Novelty" by Derek S. Prijatelj, Samuel Grieggs, Futoshi Yumoto, Eric Robertson, and Walter J. Scheirer.

This code builds training and tests based on the IAM Dataset (https://fki.tic.heia-fr.ch/databases/iam-handwriting-database).

The [manipulation code]links text](manipulations/algorithms) can introduce several types of novelties including:
(1) Pen  novelties (thickness, color, intensity)
(2) Paper/Background novelties (noise patterns, background imagery) 
(3) Style novelties (slant, spacing)
The manipulation code is used in this [notebook](notebooks/novelty_generation.ipynb).  The 

Unknown writer novelties is initially achieved using hold out sets. 
We use a [knowledge-base](data/knowledge_base) to describe the properties (line spacing, word spacing, letter size, letter slant) of each writer to form similarties between writers.
We postulate writers with similar attributes are harder to distinguish.

Given the limited number of writer samples per each writer, we create samples by creating new lines of text using sample words per each writer. 
Using the writer's stylistic distributions, the lines of text contain a representative word spacing.  The notebook to generate these new data sets is [here](notebooks/writer-identification-line-mixes-generation.ipynb).

We train a baseline writer identifier module without novelty detection [here](writer_identification/notebooks/training_writer_id.ipynb).
