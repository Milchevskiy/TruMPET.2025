Protein Secondary Structure Prediction by TruMPET 2025.

1.	Databases preparation
<br>You will need frequency extrapolation databases and the AAindex database for protein secondary structure prediction. Precomputed databases (≈7.5 GB) are available for download at:
<br>https://ftp.eimb.ru/Milch/TruMPET.2025/Databases/TruMPET2025.databases.tar.xz
<br>Unpack this archive into the Databases directory. 

2.	Default path configuration
<br>Before running any scripts, carefully check all directory paths and file names in files TruMPET_cpu.py and TruMPET_gpu.py. The default paths are the following:
```
MODEL_PATH      = "Models/mix/1024_4_cpu.pt"
TASK_SET_FILE   = "Models/mix/1024_4.task"
OUT_DIR         = "results"
LOG_LEVEL       = "INFO"
PATH_TO_FREQUENCY_STORE    = "Databases/FrequencyExtrapolation/"
PATH_TO_AAINDEX_FILE       = "Databases/AAindex/aaindex.txt"
PATH_TO_AAINDEX_TRI_LETTER = "Databases/AAindex/aaindex_mutant3.txt"
```
3.	Protein secondary structure prediction (PSSP)
<br>PSSP can be performed on CPU either on GPU. TruMPET2025 recognizes files in two formats: FASTA file – this mode performs protein secondary structure prediction without consideration of non-canonic amino acids; the input for prediction with consideration of non-canonic amino acids DATA file that must follow this structure:
<br>•	Line 1: Protein chain in three-letter amino acid codes, separated by spaces.
<br>•	Line 2: The same protein chain in one-letter amino acid codes, without spaces.
<br>•	No headers or comments are allowed in the file.
Example:
```
ALA GLY SER THR TYR
AGSTY
```
Usage examples:
```
python3 TruMPET2025_cpu.py -d 12ASA.data
python3 TruMPET2025_cuda.py -f 5B68_A.fasta
python3 TruMPET2025_cpu.py -f *.fasta
python3 TruMPET2025_cuda.py -d *.data
```

The results of PSSP are stored in the directory specified by the OUT_DIR variable (e.g., in the 'results/' subdirectory in the example above). 
