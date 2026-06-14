gene-level full-length curated training data
input_transcript_sequence_files=18
gene_level_rows=35307
genes=13371
sources=18

sample_level=gene-level/source_gene
representative_rule=one representative transcript per source_file/species/gene_name; canonical transcript if annotated, otherwise longest transcript; labels are unioned within that source-gene.
dedup_policy=Biever2020 and June2023 overlapping bulk genes are replaced by dual-location labels; non-overlapping bulk-only genes are kept as separate bulk-only files.
ouwenga_policy=uses gene_fixed dual-location full-length file; normal no-single-cell Ouwenga file is not included separately.

species_counts:
species  gene_level_rows
  mouse            27672
    rat             7346
  human              289

label_counts:
Cell_body    20931
Ribosome      5162
Neuropil      4602
Neurite       4005
Cytoplasm     3743
Synap         3452
Axon          3041
Dendrite      1522

files:
transcript_sequence_inputs/  symlinks to full-length transcript sequence CSVs
gene_level_representative_samples.csv  gene-level manifest without sequence column
stats/*.csv  label/source/species/combination summaries
training_command.sh  suggested DNABERT command
