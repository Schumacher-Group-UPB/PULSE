import json
import subprocess
import matplotlib.pyplot as plt
import sys
import matplotlib.colors as mcolors
import pandas as pd
import copy
import numpy as np


def run(cmd):
    print(cmd)
    p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)
    return p.stdout.read().decode()
def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def read(p):
    f=open(p)
    header=[]
    data=[]
    for l in f.read().splitlines():
        if l.startswith("#"):
            header=l[1:].split(";")
        else:
            d=[]
            for g in l.split(";"):
                if is_float(g):
                    d.append(float(g))
                else:
                    d.append(g)
            data.append(d)
    return {"header":header,"data":data}

def getdata(df,x,y):
    cols=[]
    ix=0
    iy=0
    for ic in range(len(df["header"])):
        if df["header"][ic]!=x and df["header"][ic]!=y:
            cols.append(ic)
        if df["header"][ic]==x:
            ix=ic
        if df["header"][ic]==y:
            iy=ic
    print(cols)
    #get unique combinations of values in cols
    unique=[]
    for d in df["data"]:
        g=[d[i] for i in cols]
        if not (g in unique):
            unique.append(g)

    D=[]
    for iu in range(len(unique)):
        u=unique[iu]
        vx=[]
        vy=[]
        for d in df["data"]:
            g=[d[i] for i in cols]
            if g==u:
                try:
                    vy.append(d[ix]*d[ix]/d[iy])
                    vx.append(d[ix])
                except:
                    pass
        g={"x":vx,"y":vy}
        for iu in range(len(cols)):
            g[df["header"][cols[iu]]]=u[iu]
        D.append(g)
    return D

def select(D,L):
    for d in D:
        ret=True
        for l in L:
            if d[l]!=L[l]:
                ret=False
        if ret:
            return d
    raise RuntimeError("set not found")

#plt.rcParams['text.usetex'] = True

col=[]
for i in mcolors.TABLEAU_COLORS:
    col.append(i)

#bandwidths in MB/s
cachebw={"7763_ccd":517223.70,"a100":5400000}
membw={"7763_ccd":38609.48,"a100":1400000}

buffer_fp32=8
buffer_fp64=16
rw={"full":23+8,"calculate_k":4*(3+1),"intermediate_sum":3*(3+1),"final_sum":1*(5+1)}

#subgrid dependency: 7763_ccd
########################################################################################
d="noctua2_7763_subgrids_ccd"
run("python bench.py -c "+d+".json -d -o "+d+".csv -f makes-label,makes-label=full_fp32,grids,subgrids-0,subgrids-1,mus/it")

df=read(d+".csv")  
D=getdata(df,"grids","mus/it")

fig = plt.figure()
fig, ax = plt.subplots(figsize=(5, 3))
ax.set_xlabel(r'grid size $N$')
ax.set_ylabel(r'grid point updates per second [MUP/s]')
#ax.set_yscale("log")
#ax.set_xlim(0,120)

s=select(D,{"makes-label":"full_fp32","subgrids-0":4,"subgrids-1":2})
ax.plot(s["x"],s["y"],linewidth=1,label="fp32, subgrid 4x2",color=col[0])

s=select(D,{"makes-label":"full_fp32","subgrids-0":4,"subgrids-1":4})
ax.plot(s["x"],s["y"],linewidth=1,label="fp32, subgrid 4x4",color=col[1])

s=select(D,{"makes-label":"full_fp32","subgrids-0":8,"subgrids-1":4})
ax.plot(s["x"],s["y"],linewidth=1,label="fp32, subgrid 8x4",color=col[2])

s=select(D,{"makes-label":"full_fp32","subgrids-0":8,"subgrids-1":8})
ax.plot(s["x"],s["y"],linewidth=1,label="fp32, subgrid 8x8",color=col[3])

#cache-bw
pcache=cachebw["7763_ccd"]/(buffer_fp32*rw["full"])
pmem=membw["7763_ccd"]/(buffer_fp32*rw["full"])

ax.hlines(pcache,0,10000,linestyles="dashed",linewidth=1,label="fp32, L3 cache bound",color=col[4])
ax.hlines(pmem,0,10000,linestyles="dotted",linewidth=1,label="fp32, mem bound",color=col[5])
ax.set_xlim(s["x"][0],s["x"][-1])

plt.legend(loc='upper right',ncol=1)
plt.tight_layout()
fig.savefig("7763_ccd_subgrids_fp32.png",dpi=600)
plt.close(fig)

#subgrid dependency: 7763_ccd
########################################################################################
fig = plt.figure()
fig, ax = plt.subplots(figsize=(5, 3))
ax.set_xlabel(r'grid size $N$')
ax.set_ylabel(r'LLC cache size [MB]')
#ax.set_yscale("log")
#ax.set_xlim(0,120)

y=[]
y2=[]
x=list(range(0,3000))
for X in x:
    y.append(X*X*buffer_fp32*6/1024/1024)
    y2.append(X*X*buffer_fp64*6/1024/1024)
ax.plot(x,y,linewidth=1,label="active data size in fp32",color=col[0])
ax.plot(x,y2,linewidth=1,label="active data size in fp64",color=col[1])

#ax.hlines(32,0,10000,linestyles="dashed",linewidth=1,label="LLC size 7763, one CCD, 8 cores",color=col[2])
ax.hlines(40,0,10000,linestyles="dashed",linewidth=1,label="LLC size NVIDIA A100",color=col[2])
ax.hlines(32,0,10000,linestyles="dashed",linewidth=1,label="LLC size NVIDIA 4060TI",color=col[3])
ax.hlines(256,0,10000,linestyles="dashed",linewidth=1,label="LLC size AMD 7763",color=col[4])
ax.hlines(96,0,10000,linestyles="dashed",linewidth=1,label="LLC size AMD 5800X3D",color=col[5])

gradient = np.linspace(0, 1, 100).reshape(1, -1)
x0=0
x1=2500
y0=0
y1=450
a = np.zeros((x1,y1))
for x in range(x1):
    for y in range(y1):
        a[x,y]=x*x*buffer_fp32*6/1024/1024*1.5-y

ax.set_xlim(x0,x1)
ax.set_ylim(y0,y1)
plt.imshow(a , extent=[x1,x0,y0,y1], aspect='auto', cmap='RdYlBu')
plt.text(50,180,'cache bandwidth bound',fontsize="7")
plt.text(1600,65,'memory bandwidth bound',fontsize="7")
#ax.set_yscale('log')
plt.legend(loc='upper left',ncol=1,fontsize="7")
plt.tight_layout()
fig.savefig("cache_sizes.png",dpi=600)
plt.close(fig)

#subgrid dependency: a100
########################################################################################
d="noctua2_a100_subgrids"
run("python bench.py -c "+d+".json -d -o "+d+".csv -f makes-label,makes-label=full_fp32,grids,subgrids-0,subgrids-1,mus/it")

df=read(d+".csv")  
D=getdata(df,"grids","mus/it")

fig = plt.figure()
fig, ax = plt.subplots(figsize=(5, 3))
ax.set_xlabel(r'grid size $N$')
ax.set_ylabel(r'grid point update rate [MUP/s]')
#ax.set_yscale("log")
#ax.set_xlim(0,120)

s=select(D,{"makes-label":"full_fp32","subgrids-0":1,"subgrids-1":1})
ax.plot(s["x"],s["y"],linewidth=1,label="fp32, subgrid 1x1",color=col[0])

s=select(D,{"makes-label":"full_fp32","subgrids-0":2,"subgrids-1":1})
ax.plot(s["x"],s["y"],linewidth=1,label="fp32, subgrid 2x1",color=col[1])

s=select(D,{"makes-label":"full_fp32","subgrids-0":4,"subgrids-1":1})
ax.plot(s["x"],s["y"],linewidth=1,label="fp32, subgrid 4x1",color=col[3])

s=select(D,{"makes-label":"full_fp32","subgrids-0":4,"subgrids-1":2})
ax.plot(s["x"],s["y"],linewidth=1,label="fp32, subgrid 4x2",color=col[5])

#cache-bw
pcache=cachebw["a100"]/(buffer_fp32*rw["full"])
pmem=membw["a100"]/(buffer_fp32*rw["full"])

ax.hlines(pcache,0,10000,linestyles="dashed",linewidth=1,label="fp32, L2 cache bound",color=col[4])
ax.hlines(pmem,0,10000,linestyles="dotted",linewidth=1,label="fp32, mem bound",color=col[5])
ax.set_xlim(s["x"][0],s["x"][-1])

plt.legend(loc='upper right',ncol=1)
plt.tight_layout()
fig.savefig("a100_subgrids_fp32.png",dpi=600)
plt.close(fig)

#individual kernel
########################################################################################
d="noctua2_7763_subgrids_ccd_kernel"
run("python bench.py -c "+d+".json -d -o "+d+".csv -f makes-label,grids,subgrids-0,subgrids-1,mus/it")

df=read(d+".csv")  
D=getdata(df,"grids","mus/it")

fig = plt.figure()
fig, ax = plt.subplots(figsize=(5, 3))
ax.set_xlabel(r'grid size $N$')
ax.set_ylabel(r'grid point update rate [MUP/s]')
#ax.set_yscale("log")
#ax.set_xlim(0,120)

s=select(D,{"makes-label":"nohalo_sync"})
ax.plot(s["x"],s["y"],linewidth=1,label="fp32, full without halo sync",color=col[0])
pcache=cachebw["7763_ccd"]/(buffer_fp32*rw["full"])
pmem=membw["7763_ccd"]/(buffer_fp32*rw["full"])
ax.hlines(pcache,0,10000,linestyles="dashed",linewidth=1,label="",color=col[0])
ax.hlines(pmem,0,10000,linestyles="dotted",linewidth=1,label="",color=col[0])
ax.set_xlim(s["x"][0],s["x"][-1])

s=select(D,{"makes-label":"calculate_k"})
ax.plot(s["x"],s["y"],linewidth=1,label="fp32, derivative",color=col[1])
pcache=cachebw["7763_ccd"]/(buffer_fp32*rw["calculate_k"])
pmem=membw["7763_ccd"]/(buffer_fp32*rw["calculate_k"])
ax.hlines(pcache,0,10000,linestyles="dashed",linewidth=1,label="",color=col[1])
ax.hlines(pmem,0,10000,linestyles="dotted",linewidth=1,label="",color=col[1])


s=select(D,{"makes-label":"intermediate_sum"})
ax.plot(s["x"],s["y"],linewidth=1,label="fp32, two-term-sum",color=col[2])
pcache=cachebw["7763_ccd"]/(buffer_fp32*rw["intermediate_sum"])
pmem=membw["7763_ccd"]/(buffer_fp32*rw["intermediate_sum"])
ax.hlines(pcache,0,10000,linestyles="dashed",linewidth=1,label="",color=col[2])
ax.hlines(pmem,0,10000,linestyles="dotted",linewidth=1,label="",color=col[2])


s=select(D,{"makes-label":"final_sum"})
ax.plot(s["x"],s["y"],linewidth=1,label="fp32, five-term-sum",color=col[3])
pcache=cachebw["7763_ccd"]/(buffer_fp32*rw["final_sum"])
pmem=membw["7763_ccd"]/(buffer_fp32*rw["final_sum"])
ax.hlines(pcache,0,10000,linestyles="dashed",linewidth=1,label="",color=col[3])
ax.hlines(pmem,0,10000,linestyles="dotted",linewidth=1,label="",color=col[3])


plt.legend(loc='upper right',ncol=1)
plt.tight_layout()
fig.savefig("7763_ccd_subgrids_fp32_kernel.png",dpi=600)
plt.close(fig)

