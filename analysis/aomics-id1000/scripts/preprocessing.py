import pandas as pd
import os

ROOT_DIR = '/Users/artemii/obrain_lab/projects/obesity-types/grant-proposals/erkko2023'
DATA_DIR = os.path.join("datasets","aomics-id1000")

participants_file = os.path.join(ROOT_DIR, DATA_DIR, "aomics_ID1000_participants.tsv")
data = pd.read_csv(participants_file, sep="\t")

print(data.describe())