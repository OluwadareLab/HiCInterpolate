import os
import traceback
import numpy as np
from downstream_analysis.HiCGNN import hicgnn
EPOCHS = 2
HICGNN_FILENAME = "./hicgnn/hicgnn.py"


def run_hicgnn(input, output, start, end):
    try:
        _, ext = os.path.splitext(input)
        print("Extension:", ext)
        if ext.lower() == ".npy":
            full_matrix = np.load(input)
        if ext.lower() == ".txt":
            full_matrix = np.loadtxt(input)

        region = [start, end]
        matrix = full_matrix[region[0]:region[1], region[0]:region[1]]
        matrix_file = os.path.join(output, f"{region[0]}_{region[1]}.txt")
        np.savetxt(matrix_file, matrix, fmt="%.6f")

        print(f"Processing {matrix_file}")
        hicgnn.hicgnn(matrix_file, EPOCHS, output)
        # cmd = [
        #     sys.executable,
        #     HICGNN_FILENAME,
        #     f"{matrix_file}",
        #     "-o", f"{output}",
        #     "-ep", str(EPOCHS)
        # ]
        
        # subprocess.run(cmd, check=True)
        print(f"Completed {matrix_file}")
    except Exception as ex:
        print(f"Exception\n: {ex}")
        traceback.print_exc()
