import pandas as pd
import sys
import gdown

input_file, output_file = sys.argv[1:3]
gdrive_url = "https://drive.google.com/uc?id=1ROpZ5jfrpJSp5T3-b8bjZX0MDLDCHMTO"
gdown.download(gdrive_url, input_file, quiet=False)
df = pd.read_csv(input_file, delimiter='\t', quoting=3)
df.to_csv(output_file, sep='\t', index=False, quoting=3)