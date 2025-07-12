import numpy as np
import glob

def print_npy_files_content():
    npy_files = glob.glob('*.npy')

    if not npy_files:
        print("No .npy files found in the current directory.")
        return

    for file_path in npy_files:
        print(f"\n--- Contents of {file_path} ---")
        try:
            data = np.load(file_path)
            print(data)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

if __name__ == "__main__":
    print_npy_files_content()
