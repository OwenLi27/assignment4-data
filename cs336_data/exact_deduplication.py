import os
import mmh3
from collections import defaultdict

def exact_deduplication(input_files: list[os.PathLike], output_directory: os.PathLike):
    """
    Perform exact line deduplication on the input files and save the results to the output directory.

    Args:
        input_files (list[os.PathLike]): List of input file paths.
        output_directory (os.PathLike): Path to the output directory where deduplicated files will be saved.
    """
    unique_lines = defaultdict(int)

    # Iterate through each input file
    for input_file in input_files:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                unique_lines[mmh3.hash128(line)] += 1

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # write files with same naming to output directory 
    for input_file in input_files:
        output_file_path = os.path.join(output_directory, os.path.basename(input_file))
        with open(output_file_path, 'w', encoding='utf-8') as out_f:
            with open(input_file, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    if unique_lines[mmh3.hash128(line)] == 1:
                        out_f.write(line)
