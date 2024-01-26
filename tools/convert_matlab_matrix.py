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

    # Convert the matlab file to PC3
    x,y = np.linspace(-float(xmax), float(xmax), N), np.linspace(-float(ymax), float(ymax), N)
    z = mat[name].flatten()
    print(f"Minimum value: {np.min(z)}, Maximum value: {np.max(z)}")
    return x,y,z,N

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to the matlab file")
    parser.add_argument("-complex", action="store_true", help="Whether to convert the complex part of the variable")
    parser.add_argument("-rotate", action="store_true", help="Rotate the Matrix by 90 degrees")
    # Add x,y arguments in a single --dim argument
    parser.add_argument("--dim", type=str, required=False, help="Xmax, Ymax", default="-1,1")
    args = parser.parse_args()

    # Load the matlab file
    x,y,z,N = load_matrix(args.file, "u", args.dim, args.complex, args.rotate)

    # Save the PC3 file
    destination = args.file.replace(".mat", ".txt")
    with open(destination, "w") as f:
        for i in range(N):
            for j in range(N):
                index = i*N+j if not args.rotate else j*N+i
                f.write(f"{z[index].real} " if args.complex else f"{z[index]} ")
            f.write("\n")
        if args.complex:
            for i in range(N):
                for j in range(N):
                    index = i*N+j if not args.rotate else j*N+i
                    f.write(f"{z[index].complex} "
            f.write("\n")