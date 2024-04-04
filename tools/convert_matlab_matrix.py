import numpy as np
import scipy.io as sio
import argparse

"""
-----------------------------------------------------------------------------------

                  _____    _     _            _______   _______
                 |_____]   |     |   |        |______   |______
                 |       . |_____| . |_____ . ______| . |______ .

        Paderborn Ultrafast SoLver for the nonlinear Schroedinger Equation

-----------------------------------------------------------------------------------

Conversion Tool for Matlab matrices

Usage:
    python convert_matlab_matrix.py --file <path_to_file> [-complex] [-rotate] [--dim <x,y>]
"""




def load_matrix(filename: str, name: str, dim: str, complex: bool = False, rotate: bool = False) -> tuple:
    """
    Loads a Matlab matrix and converts it to P.U.L.S.E. format.
    :param filename: Path to the matlab file
    :param name: Name of the variable to load
    :param dim: Dimensions of the matrix
    :param complex: Whether to convert the complex part of the variable
    :param rotate: Rotate the Matrix by 90 degrees
    """

    # Load the matlab file
    mat = sio.loadmat(args.file)
    name = [ name for name in mat.keys() if not name.startswith("__") ][0]

    print(f"Loading .mat of shape {mat[name].shape}")
    N = np.sqrt(np.prod(mat[name].shape)).astype(int)

    # Dimensions
    xmax,ymax = args.dim.split(",") if "," in args.dim else (args.dim,args.dim)
    dx,dy = float(xmax)/N, float(ymax)/N

    # Convert the matlab file to PC3
    z = mat[name].flatten()
    print(f"Minimum value: {np.min(z)}, Maximum value: {np.max(z)}")
    return z,N,xmax,ymax,dx,dy

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to the matlab file")
    parser.add_argument("-complex", action="store_true", help="Whether to convert the complex part of the variable")
    parser.add_argument("-rotate", action="store_true", help="Rotate the Matrix by 90 degrees")
    # Add x,y arguments in a single --dim argument
    parser.add_argument("--dim", type=str, required=False, help="Xmax, Ymax", default="-1,1")
    parser.add_argument("--out", type=str, required=False, help="Source file name with .txt extension.", default=None)
    args = parser.parse_args()

    # Load the matlab file
    z,N,xmax,ymax,dx,dy = load_matrix(args.file, "u", args.dim, args.complex, args.rotate)

    # if N is uneven, we will add a zero at the end of each row and column.
    uneven = False
    if N % 2 != 0:
        uneven = True
        print("PULSE requireds even matrices. Padding matrix from {} to {}".format(N, N+1))

    # Save the PC3 file
    destination = args.file.replace(".mat", ".txt") if args.out is None else args.out
    with open(destination, "w") as f:
        # Write Header
        size = N+1 if uneven else N
        f.write(f"# SIZE {size} {size} {xmax} {ymax} {dx} {dy} :: PYTHON-GENERATED\n")
        for i in range(N):
            for j in range(N):
                index = i*N+j if not args.rotate else j*N+i
                f.write(f"{z[index].real} " if args.complex else f"{z[index]} ")
            if uneven:
                f.write("0 ")
            f.write("\n")
        if uneven:
            f.write("0 "*size+"\n")
        if args.complex:
            for i in range(N):
                for j in range(N):
                    index = i*N+j if not args.rotate else j*N+i
                    f.write(f"{z[index].imag} ")
                if uneven:
                    f.write("0 ")
                f.write("\n")
            if uneven:
                f.write("0 "*size+"\n")