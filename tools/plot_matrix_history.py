import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import os
import multiprocessing

"""
-----------------------------------------------------------------------------------

                  _____    _     _            _______   _______
                 |_____]   |     |   |        |______   |______
                 |       . |_____| . |_____ . ______| . |______ .

        Paderborn Ultrafast SoLver for the nonlinear Schroedinger Equation

-----------------------------------------------------------------------------------

Plotting Tool for Pulse Matrix History Output

Usage:
    python plot_matrix_history.py --file <path_to_file> [-phase] [--fps <fps>]
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to one of the .txt files to plot.")
    parser.add_argument("-noframe", action="store_true", help="Disable Axis and Colorbar.")
    parser.add_argument("-fixedrange", action="store_true", help="Fixes the min and max to the final min/max.")
    parser.add_argument("--fps", type=int, required=False, help="FPS.", default=50)
    parser.add_argument("--skip", type=int, required=False, help="Skip.", default=1)
    parser.add_argument("--colormap", type=str, required=False, help="Colormap.", default="hot")
    parser.add_argument("--func", type=str, required=False, help="Lambda to apply to each element as lambda mat: func(mat).", default="np.abs(mat)")
    parser.add_argument("--suffix", type=str, required=False, help="File Suffix.", default="")

    args = parser.parse_args()

    path, file = os.path.dirname(args.file), "_".join(os.path.basename(args.file).split("_")[:-1])

    files = [f for f in os.listdir(path) if f.endswith('.txt') and file in f]

    files.sort(key=lambda x: float(x.split('_')[-1].replace(".txt", "")))

    N_files = len(files)//args.skip

    # Create animation of files. Plot every single file using contourf.
    lambdas_ = [eval("lambda mat: "+func, { "np" : np }) for func in args.func.split(";")]
    
    fig,axes = plt.subplots( 1, len(lambdas_), figsize=(5*len(lambdas_),5) )

    # Find maximum and minimum values
    if args.fixedrange:
        mmax, mmin = -1E10, 1E10
        for current in files:
            x,y,re,im = np.loadtxt(os.path.join(path,current), unpack=True)
            mmax = max(mmax, max( np.max(re),np.max(im)))
            mmin = min(mmin, min( np.min(re),np.min(im)))
    else:
        mmax, mmin = None, None
    print(f"Maximum: {mmax}, Minimum: {mmin}")

    def anim(i):
        index = i*args.skip
        x,y,re,im = np.loadtxt(os.path.join(path,files[index]), unpack=True)
        for ax,lambda_ in zip(axes.flatten(),lambdas_):
            ax.clear()
            try:
                z = lambda_(re+im*1j)
                ax.tricontourf(x,y,z,levels=30,cmap=args.colormap,vmin=mmin,vmax=mmax)
                # Colorbar
                if args.noframe:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_frame_on(False)
                ax.set_aspect('equal')
                ax.set_title(f"t = {files[index].split('_')[-1].replace('.txt','')} ps")
            except Exception as e:
                print(f"Error plotting {files[index]}: {e}")
                pass
        print(f"Done plotting {files[index]} ({i+1}/{len(files)})",end="\r")

    ani = animation.FuncAnimation(fig, anim, frames=N_files, interval=1000.0/args.fps, repeat_delay=1000)
    filename = os.path.join(path,file+args.suffix+".mp4")
    ani.save(filename, writer='ffmpeg')