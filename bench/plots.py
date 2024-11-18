import json
import subprocess
import matplotlib.pyplot as plt
import sys
import matplotlib.colors as mcolors
import pandas as pd
import copy
import numpy as np
from matplotlib.ticker import (MultipleLocator, 
                                       FormatStrFormatter, 
                                                                      AutoMinorLocator) 

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
                    vy.append(d[ix]*d[ix]/d[iy]/1000)
                    vx.append(d[ix])
                except:
                    pass
        g={"x":vx,"y":vy}
        for iu in range(len(cols)):
            g[df["header"][cols[iu]]]=u[iu]
        D.append(g)
    return D

def getdatadiv(df,x,y,z):
    cols=[]
    ix=0
    iy=0
    iz=0
    for ic in range(len(df["header"])):
        if df["header"][ic]!=x and df["header"][ic]!=y and df["header"][ic]!=z:
            cols.append(ic)
        if df["header"][ic]==x:
            ix=ic
        if df["header"][ic]==y:
            iy=ic
        if df["header"][ic]==z:
            iz=ic
    print(cols)
    #get unique combinations of values in cols
    unique=[]
    for d in df["data"]:
        g=[d[i] for i in cols]
        if not (g in unique):
            unique.append(g)

    D=[]
    print("unique=",len(unique),unique)
    for iu in range(len(unique)):
        u=unique[iu]
        vx=[]
        vy=[]
        for d in df["data"]:
            g=[d[i] for i in cols]
            if g==u:
                try:
                    vy.append((d[iz]/10.0)/(d[ix]*d[ix]/d[iy]/1000))
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
    raise RuntimeError("set not found "+str(L))

def selectmin(D,L):
    dmin={}
    for d in D:
        ret=True
        for l in L:
            if d[l]!=L[l]:
                ret=False
        if ret:
            for i in range(len(d["x"])):
                if not (d["x"][i] in dmin):
                    dmin[d["x"][i]]=d["y"][i]
                else:
                    dmin[d["x"][i]]=min(dmin[d["x"][i]],d["y"][i])
    d={"x":[],"y":[]}
    for f in sorted(dmin.keys()):
        d["x"].append(f)
        d["y"].append(dmin[f])
    return d
    raise RuntimeError("set not found "+str(L))

#plt.rcParams['text.usetex'] = True

col=[]
for i in mcolors.TABLEAU_COLORS:
    col.append(i)

#bandwidths in MB/s
cachebw={"7763_ccd":517.22370,"a100":5400.000,"7763_full":2600.000,"5800X3D":580,"4060ti":1600}
membw={"7763_ccd":38.60948,"a100":1400.000,"7763_full":155.000,"5800X3D":31,"4060ti":275}
fp={"4060ti":0.3,"5800X3D":0.5}

buffer_fp32=8
buffer_fp64=16
rw={"full":23+8,"calculate_k":4*(3+1),"intermediate_sum":3*(2+1),"final_sum":1*(5+1)}

#cache plot
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

#subgrid dependency: 7763_ccd
########################################################################################
d="noctua2_7763_subgrids_ccd"
run("python bench.py -c "+d+".json -d -o "+d+".csv -f makes-label,makes-label=full_fp32,grids,subgrids-0,subgrids-1,mus/it")

df=read(d+".csv")  
D=getdata(df,"grids","mus/it")

fig = plt.figure()
fig, ax = plt.subplots(figsize=(3, 3))
ax.set_xlabel(r'grid size $N$')
ax.set_ylabel(r'grid point update rate [GUP/s]')
plt.text(200,1.6,'AMD 7763\none CCD\nfp32',fontsize="8")
#ax.set_yscale("log")
#ax.set_xlim(0,120)
#[[2,4],[4,2],[4,4],[4,6],[8,4],[8,8],[16,8],[16,16],[16,32],[32,32]],
s=select(D,{"makes-label":"full_fp32","subgrids-0":2,"subgrids-1":4})
ax.plot(s["x"],s["y"],linewidth=1,label="2x4",color=col[0])

s=select(D,{"makes-label":"full_fp32","subgrids-0":4,"subgrids-1":4})
ax.plot(s["x"],s["y"],linewidth=1,label="4x4",color=col[1])

s=select(D,{"makes-label":"full_fp32","subgrids-0":8,"subgrids-1":4})
ax.plot(s["x"],s["y"],linewidth=1,label="8x4",color=col[2])

s=select(D,{"makes-label":"full_fp32","subgrids-0":8,"subgrids-1":8})
ax.plot(s["x"],s["y"],linewidth=1,label="8x8",color=col[3])

s=select(D,{"makes-label":"full_fp32","subgrids-0":16,"subgrids-1":8})
ax.plot(s["x"],s["y"],linewidth=1,label="16x8",color=col[4])

#cache-bw
pcache=cachebw["7763_ccd"]/(buffer_fp32*rw["full"])
pmem=membw["7763_ccd"]/(buffer_fp32*rw["full"])

ax.hlines(pcache,0,10000,linestyles="dashed",linewidth=1,label="L3 cache bound",color="black")
ax.hlines(pmem,0,10000,linestyles="dotted",linewidth=1,label="mem bound",color="black")

ax.set_xlim(s["x"][0],s["x"][-1])
ax.set_ylim(0,pcache+0.1)
ax.yaxis.set_major_locator(MultipleLocator(1)) 
ax.yaxis.set_major_formatter(FormatStrFormatter(' % 1.0f')) 

plt.legend(loc='upper right',ncol=1,fontsize=8)
plt.tight_layout()
fig.savefig("7763_ccd_subgrids_fp32.png",dpi=600)
plt.close(fig)


#subgrid dependency: 7763_ccd
########################################################################################
d="noctua2_7763_subgrids_ccd"
run("python bench.py -c "+d+".json -d -o "+d+".csv -f makes-label,makes-label=full_fp64,grids,subgrids-0,subgrids-1,mus/it")

df=read(d+".csv")  
D=getdata(df,"grids","mus/it")

fig = plt.figure()
fig, ax = plt.subplots(figsize=(3, 3))
ax.set_xlabel(r'grid size $N$')
ax.set_ylabel(r'grid point update rate [GUP/s]')
plt.text(200,0.8,'AMD 7763\none CCD\nfp64',fontsize="8")
#ax.set_yscale("log")
#ax.set_xlim(0,120)

s=select(D,{"makes-label":"full_fp64","subgrids-0":2,"subgrids-1":4})
ax.plot(s["x"],s["y"],linewidth=1,label="2x4",color=col[0])

s=select(D,{"makes-label":"full_fp64","subgrids-0":4,"subgrids-1":4})
ax.plot(s["x"],s["y"],linewidth=1,label="4x4",color=col[1])

s=select(D,{"makes-label":"full_fp64","subgrids-0":8,"subgrids-1":4})
ax.plot(s["x"],s["y"],linewidth=1,label="8x4",color=col[2])

s=select(D,{"makes-label":"full_fp64","subgrids-0":8,"subgrids-1":8})
ax.plot(s["x"],s["y"],linewidth=1,label="8x8",color=col[3])

s=select(D,{"makes-label":"full_fp64","subgrids-0":16,"subgrids-1":8})
ax.plot(s["x"],s["y"],linewidth=1,label="16x8",color=col[4])

#cache-bw
pcache=cachebw["7763_ccd"]/(buffer_fp64*rw["full"])
pmem=membw["7763_ccd"]/(buffer_fp64*rw["full"])

ax.hlines(pcache,0,10000,linestyles="dashed",linewidth=1,label="L3 cache bound",color="black")
ax.hlines(pmem,0,10000,linestyles="dotted",linewidth=1,label="mem bound",color="black")

ax.set_xlim(s["x"][0],s["x"][-1])
ax.set_ylim(0,pcache+0.1)
ax.yaxis.set_major_locator(MultipleLocator(1)) 
ax.yaxis.set_major_formatter(FormatStrFormatter(' % 1.0f')) 

plt.legend(loc='upper right',ncol=1,fontsize=8)
plt.tight_layout()
fig.savefig("7763_ccd_subgrids_fp64.png",dpi=600)
plt.close(fig)

#subgrid dependency: 7763_full
########################################################################################
d="noctua2_7763_subgrids_full"
run("python bench.py -c "+d+".json -d -o "+d+".csv -f makes-label,makes-label=full_fp32,grids,subgrids-0,subgrids-1,mus/it")

df=read(d+".csv")  
D=getdata(df,"grids","mus/it")

fig = plt.figure()
fig, ax = plt.subplots(figsize=(3, 3))
ax.set_xlabel(r'grid size $N$')
ax.set_ylabel(r'grid point update rate [GUP/s]')
plt.text(200,9,'AMD 7763\nfp32',fontsize="8")
#ax.set_yscale("log")
#ax.set_xlim(0,120)

s=select(D,{"makes-label":"full_fp32","subgrids-0":8,"subgrids-1":8})
ax.plot(s["x"],s["y"],linewidth=1,label="8x8",color=col[0])

s=select(D,{"makes-label":"full_fp32","subgrids-0":16,"subgrids-1":8})
ax.plot(s["x"],s["y"],linewidth=1,label="16x8",color=col[1])

s=select(D,{"makes-label":"full_fp32","subgrids-0":16,"subgrids-1":16})
ax.plot(s["x"],s["y"],linewidth=1,label="16x16",color=col[2])

s=select(D,{"makes-label":"full_fp32","subgrids-0":16,"subgrids-1":32})
ax.plot(s["x"],s["y"],linewidth=1,label="16x32",color=col[3])

s=select(D,{"makes-label":"full_fp32","subgrids-0":32,"subgrids-1":32})
ax.plot(s["x"],s["y"],linewidth=1,label="32x32",color=col[4])


#cache-bw
pcache=cachebw["7763_full"]/(buffer_fp32*rw["full"])
pmem=membw["7763_full"]/(buffer_fp32*rw["full"])

ax.hlines(pcache,0,10000,linestyles="dashed",linewidth=1,label="L3 cache bound",color="black")
ax.hlines(pmem,0,10000,linestyles="dotted",linewidth=1,label="mem bound",color="black")

ax.set_xlim(s["x"][0],s["x"][-1])
ax.set_ylim(0,pcache+0.1)

plt.legend(loc='upper right',ncol=1,fontsize=8)
plt.tight_layout()
fig.savefig("7763_full_subgrids_fp32.png",dpi=600)
plt.close(fig)


#subgrid dependency: 7763_full
########################################################################################
d="noctua2_7763_subgrids_full"
run("python bench.py -c "+d+".json -d -o "+d+".csv -f makes-label,makes-label=full_fp64,grids,subgrids-0,subgrids-1,mus/it")

df=read(d+".csv")  
D=getdata(df,"grids","mus/it")

fig = plt.figure()
fig, ax = plt.subplots(figsize=(3, 3))
ax.set_xlabel(r'grid size $N$')
ax.set_ylabel(r'grid point update rate [GUP/s]')
plt.text(200,4.5,'AMD 7763\nfp64',fontsize="8")
#ax.set_yscale("log")
#ax.set_xlim(0,120)

s=select(D,{"makes-label":"full_fp64","subgrids-0":8,"subgrids-1":8})
ax.plot(s["x"],s["y"],linewidth=1,label="8x8",color=col[0])

s=select(D,{"makes-label":"full_fp64","subgrids-0":16,"subgrids-1":8})
ax.plot(s["x"],s["y"],linewidth=1,label="16x8",color=col[1])

s=select(D,{"makes-label":"full_fp64","subgrids-0":16,"subgrids-1":16})
ax.plot(s["x"],s["y"],linewidth=1,label="16x16",color=col[2])

s=select(D,{"makes-label":"full_fp64","subgrids-0":16,"subgrids-1":32})
ax.plot(s["x"],s["y"],linewidth=1,label="16x32",color=col[3])

s=select(D,{"makes-label":"full_fp64","subgrids-0":32,"subgrids-1":32})
ax.plot(s["x"],s["y"],linewidth=1,label="32x32",color=col[4])


#cache-bw
pcache=cachebw["7763_full"]/(buffer_fp64*rw["full"])
pmem=membw["7763_full"]/(buffer_fp64*rw["full"])

ax.hlines(pcache,0,10000,linestyles="dashed",linewidth=1,label="L3 cache bound",color="black")
ax.hlines(pmem,0,10000,linestyles="dotted",linewidth=1,label="mem bound",color="black")

ax.set_xlim(s["x"][0],s["x"][-1])
ax.set_ylim(0,pcache+0.1)
ax.yaxis.set_major_formatter(FormatStrFormatter(' % 1.0f')) 

plt.legend(loc='upper right',ncol=1,fontsize=8)
plt.tight_layout()
fig.savefig("7763_full_subgrids_fp64.png",dpi=600)
plt.close(fig)

#subgrid dependency: 5800X3d
########################################################################################
d="5800X3D_subgrids_socket"
run("python bench.py -c "+d+".json -d -o "+d+".csv -f makes-label,makes-label=full_fp32,grids,subgrids-0,subgrids-1,mus/it")

df=read(d+".csv")  
D=getdata(df,"grids","mus/it")

fig = plt.figure()
fig, ax = plt.subplots(figsize=(3, 3))
ax.set_xlabel(r'grid size $N$')
ax.set_ylabel(r'grid point update rate [GUP/s]')
plt.text(200,1.8,'AMD\n5800X3D\nfp32',fontsize="8")
#ax.set_yscale("log")
#ax.set_xlim(0,120)
#[[2,4],[4,2],[4,4],[4,6],[8,4],[8,8],[16,8],[16,16],[16,32],[32,32]],
s=select(D,{"makes-label":"full_fp32","subgrids-0":2,"subgrids-1":4})
ax.plot(s["x"],s["y"],linewidth=1,label="2x4",color=col[0])

s=select(D,{"makes-label":"full_fp32","subgrids-0":4,"subgrids-1":4})
ax.plot(s["x"],s["y"],linewidth=1,label="4x4",color=col[1])

s=select(D,{"makes-label":"full_fp32","subgrids-0":8,"subgrids-1":4})
ax.plot(s["x"],s["y"],linewidth=1,label="8x4",color=col[2])

s=select(D,{"makes-label":"full_fp32","subgrids-0":8,"subgrids-1":8})
ax.plot(s["x"],s["y"],linewidth=1,label="8x8",color=col[3])

s=select(D,{"makes-label":"full_fp32","subgrids-0":16,"subgrids-1":8})
ax.plot(s["x"],s["y"],linewidth=1,label="16x8",color=col[4])

#cache-bw
pcache=cachebw["5800X3D"]/(buffer_fp32*rw["full"])
pmem=membw["5800X3D"]/(buffer_fp32*rw["full"])

ax.hlines(pcache,0,10000,linestyles="dashed",linewidth=1,label="L3 cache bound",color="black")
ax.hlines(pmem,0,10000,linestyles="dotted",linewidth=1,label="mem bound",color="black")

ax.set_xlim(s["x"][0],s["x"][-1])
ax.set_ylim(0,pcache+0.1)
ax.yaxis.set_major_locator(MultipleLocator(1)) 
ax.yaxis.set_major_formatter(FormatStrFormatter(' % 1.0f')) 

plt.legend(loc='upper right',ncol=1,fontsize=8)
plt.tight_layout()
fig.savefig("5800X3D_subgrids_fp32.png",dpi=600)
plt.close(fig)


#subgrid dependency: 5800X3D
########################################################################################
d="5800X3D_subgrids_socket"
run("python bench.py -c "+d+".json -d -o "+d+".csv -f makes-label,makes-label=full_fp64,grids,subgrids-0,subgrids-1,mus/it")

df=read(d+".csv")  
D=getdata(df,"grids","mus/it")

fig = plt.figure()
fig, ax = plt.subplots(figsize=(3, 3))
ax.set_xlabel(r'grid size $N$')
ax.set_ylabel(r'grid point update rate [GUP/s]')
plt.text(200,0.9,'AMD\n5800X3D\nfp64',fontsize="8")
#ax.set_yscale("log")
#ax.set_xlim(0,120)

s=select(D,{"makes-label":"full_fp64","subgrids-0":2,"subgrids-1":4})
ax.plot(s["x"],s["y"],linewidth=1,label="2x4",color=col[0])

s=select(D,{"makes-label":"full_fp64","subgrids-0":4,"subgrids-1":4})
ax.plot(s["x"],s["y"],linewidth=1,label="4x4",color=col[1])

s=select(D,{"makes-label":"full_fp64","subgrids-0":8,"subgrids-1":4})
ax.plot(s["x"],s["y"],linewidth=1,label="8x4",color=col[2])

s=select(D,{"makes-label":"full_fp64","subgrids-0":8,"subgrids-1":8})
ax.plot(s["x"],s["y"],linewidth=1,label="8x8",color=col[3])

s=select(D,{"makes-label":"full_fp64","subgrids-0":16,"subgrids-1":8})
ax.plot(s["x"],s["y"],linewidth=1,label="16x8",color=col[4])

#cache-bw
pcache=cachebw["5800X3D"]/(buffer_fp64*rw["full"])
pmem=membw["5800X3D"]/(buffer_fp64*rw["full"])

ax.hlines(pcache,0,10000,linestyles="dashed",linewidth=1,label="L3 cache bound",color="black")
ax.hlines(pmem,0,10000,linestyles="dotted",linewidth=1,label="mem bound",color="black")

ax.set_xlim(s["x"][0],s["x"][-1])
ax.set_ylim(0,pcache+0.1)
ax.yaxis.set_major_locator(MultipleLocator(1)) 
ax.yaxis.set_major_formatter(FormatStrFormatter(' % 1.0f')) 

plt.legend(loc='upper right',ncol=1,fontsize=8)
plt.tight_layout()
fig.savefig("5800X3D_subgrids_fp64.png",dpi=600)
plt.close(fig)


#subgrid dependency: a100
########################################################################################
d="noctua2_a100_subgrids"
run("python bench.py -c "+d+".json -d -o "+d+".csv -f makes-label,makes-label=full_fp32,grids,subgrids-0,subgrids-1,mus/it")

df=read(d+".csv")  
D=getdata(df,"grids","mus/it")

fig = plt.figure()
fig, ax = plt.subplots(figsize=(3, 3))
ax.set_xlabel(r'grid size $N$')
ax.set_ylabel(r'grid point update rate [GUP/s]')
plt.text(400,19,'NVIDIA A100\nfp32',fontsize="8")
#ax.set_yscale("log")
#ax.set_xlim(0,120)

s=select(D,{"makes-label":"full_fp32","subgrids-0":1,"subgrids-1":1})
ax.plot(s["x"],s["y"],linewidth=1,label="1x1",color=col[0])

s=select(D,{"makes-label":"full_fp32","subgrids-0":2,"subgrids-1":1})
ax.plot(s["x"],s["y"],linewidth=1,label="2x1",color=col[1])

s=select(D,{"makes-label":"full_fp32","subgrids-0":4,"subgrids-1":1})
ax.plot(s["x"],s["y"],linewidth=1,label="4x1",color=col[2])

s=select(D,{"makes-label":"full_fp32","subgrids-0":8,"subgrids-1":1})
ax.plot(s["x"],s["y"],linewidth=1,label="8x1",color=col[3])

s=select(D,{"makes-label":"full_fp32","subgrids-0":10,"subgrids-1":1})
ax.plot(s["x"],s["y"],linewidth=1,label="10x1",color=col[4])

#cache-bw
pcache=cachebw["a100"]/(buffer_fp32*rw["full"])
pmem=membw["a100"]/(buffer_fp32*rw["full"])

ax.hlines(pcache,0,10000,linestyles="dashed",linewidth=1,label="L2 cache bound",color="black")
ax.hlines(pmem,0,10000,linestyles="dotted",linewidth=1,label="mem bound",color="black")
ax.set_xlim(s["x"][0],s["x"][-1])
ax.set_ylim(0,pcache+0.1)

plt.legend(loc='upper right',ncol=1,fontsize=8)
plt.tight_layout()
fig.savefig("a100_subgrids_fp32.png",dpi=600)
plt.close(fig)

#subgrid dependency: a100
########################################################################################
d="noctua2_a100_subgrids"
run("python bench.py -c "+d+".json -d -o "+d+".csv -f makes-label,makes-label=full_fp64,grids,subgrids-0,subgrids-1,mus/it")

df=read(d+".csv")  
D=getdata(df,"grids","mus/it")

fig = plt.figure()
fig, ax = plt.subplots(figsize=(3, 3))
ax.set_xlabel(r'grid size $N$')
ax.set_ylabel(r'grid point update rate [GUP/s]')
plt.text(400,9.5,'NVIDIA A100\nfp64',fontsize="8")

#ax.set_yscale("log")
#ax.set_xlim(0,120)

s=select(D,{"makes-label":"full_fp64","subgrids-0":1,"subgrids-1":1})
ax.plot(s["x"],s["y"],linewidth=1,label="1x1",color=col[0])

s=select(D,{"makes-label":"full_fp64","subgrids-0":2,"subgrids-1":1})
ax.plot(s["x"],s["y"],linewidth=1,label="2x1",color=col[1])

s=select(D,{"makes-label":"full_fp64","subgrids-0":4,"subgrids-1":1})
ax.plot(s["x"],s["y"],linewidth=1,label="4x1",color=col[2])

s=select(D,{"makes-label":"full_fp64","subgrids-0":8,"subgrids-1":1})
ax.plot(s["x"],s["y"],linewidth=1,label="8x1",color=col[3])

s=select(D,{"makes-label":"full_fp64","subgrids-0":10,"subgrids-1":1})
ax.plot(s["x"],s["y"],linewidth=1,label="10x1",color=col[4])

#cache-bw
pcache=cachebw["a100"]/(buffer_fp64*rw["full"])
pmem=membw["a100"]/(buffer_fp64*rw["full"])

ax.hlines(pcache,0,10000,linestyles="dashed",linewidth=1,label="L2 cache bound",color="black")
ax.hlines(pmem,0,10000,linestyles="dotted",linewidth=1,label="mem bound",color="black")
ax.set_xlim(s["x"][0],s["x"][-1])
ax.set_ylim(0,pcache+0.1)

plt.legend(loc='upper right',ncol=1,fontsize=8)
plt.tight_layout()
fig.savefig("a100_subgrids_fp64.png",dpi=600)
plt.close(fig)

#subgrid dependency: 4060TI
########################################################################################
d="4060ti_subgrids"
run("python bench.py -c "+d+".json -d -o "+d+".csv -f makes-label,makes-label=full_fp32,grids,subgrids-0,subgrids-1,mus/it")

df=read(d+".csv")  
D=getdata(df,"grids","mus/it")

fig = plt.figure()
fig, ax = plt.subplots(figsize=(3, 3))
ax.set_xlabel(r'grid size $N$')
ax.set_ylabel(r'grid point update rate [GUP/s]')
plt.text(400,6.5,'NVIDIA 4060TI\nfp32',fontsize="8")
#ax.set_yscale("log")
#ax.set_xlim(0,120)

s=select(D,{"makes-label":"full_fp32","subgrids-0":1,"subgrids-1":1})
ax.plot(s["x"],s["y"],linewidth=1,label="1x1",color=col[0])

s=select(D,{"makes-label":"full_fp32","subgrids-0":2,"subgrids-1":1})
ax.plot(s["x"],s["y"],linewidth=1,label="2x1",color=col[1])

s=select(D,{"makes-label":"full_fp32","subgrids-0":4,"subgrids-1":1})
ax.plot(s["x"],s["y"],linewidth=1,label="4x1",color=col[2])

s=select(D,{"makes-label":"full_fp32","subgrids-0":8,"subgrids-1":1})
ax.plot(s["x"],s["y"],linewidth=1,label="8x1",color=col[3])

s=select(D,{"makes-label":"full_fp32","subgrids-0":10,"subgrids-1":1})
ax.plot(s["x"],s["y"],linewidth=1,label="10x1",color=col[4])

#cache-bw
pcache=cachebw["4060ti"]/(buffer_fp32*rw["full"])
pmem=membw["4060ti"]/(buffer_fp32*rw["full"])

ax.hlines(pcache,0,10000,linestyles="dashed",linewidth=1,label="L2 cache bound",color="black")
ax.hlines(pmem,0,10000,linestyles="dotted",linewidth=1,label="mem bound",color="black")
ax.set_xlim(s["x"][0],s["x"][-1])
ax.set_ylim(0,pcache+1.1)
ax.yaxis.set_major_locator(MultipleLocator(1)) 
ax.yaxis.set_major_formatter(FormatStrFormatter(' % 1.0f')) 

plt.legend(loc='upper right',ncol=1,fontsize=8)
plt.tight_layout()
fig.savefig("4060ti_subgrids_fp32.png",dpi=600)
plt.close(fig)

#subgrid dependency: 4060ti
########################################################################################
d="4060ti_subgrids"
run("python bench.py -c "+d+".json -d -o "+d+".csv -f makes-label,makes-label=full_fp64,grids,subgrids-0,subgrids-1,mus/it")

df=read(d+".csv")  
D=getdata(df,"grids","mus/it")

fig = plt.figure()
fig, ax = plt.subplots(figsize=(3, 3))
ax.set_xlabel(r'grid size $N$')
ax.set_ylabel(r'grid point update rate [GUP/s]')
plt.text(400,2.7,'NVIDIA 4060TI\nfp64',fontsize="8")

#ax.set_yscale("log")
#ax.set_xlim(0,120)

s=select(D,{"makes-label":"full_fp64","subgrids-0":1,"subgrids-1":1})
ax.plot(s["x"],s["y"],linewidth=1,label="1x1",color=col[0])

s=select(D,{"makes-label":"full_fp64","subgrids-0":2,"subgrids-1":1})
ax.plot(s["x"],s["y"],linewidth=1,label="2x1",color=col[1])

s=select(D,{"makes-label":"full_fp64","subgrids-0":4,"subgrids-1":1})
ax.plot(s["x"],s["y"],linewidth=1,label="4x1",color=col[2])

s=select(D,{"makes-label":"full_fp64","subgrids-0":8,"subgrids-1":1})
ax.plot(s["x"],s["y"],linewidth=1,label="8x1",color=col[3])

s=select(D,{"makes-label":"full_fp64","subgrids-0":10,"subgrids-1":1})
ax.plot(s["x"],s["y"],linewidth=1,label="10x1",color=col[4])

#cache-bw
pcache=cachebw["4060ti"]/(buffer_fp64*rw["full"])
pmem=membw["4060ti"]/(buffer_fp64*rw["full"])
pfp=fp["4060ti"]/(buffer_fp64*rw["full"]*0.333)*1000
print("fp bound",pfp)
ax.hlines(pcache,0,10000,linestyles="dashed",linewidth=1,label="L2 cache bound",color="black")
ax.hlines(pmem,0,10000,linestyles="dotted",linewidth=1,label="mem bound",color="black")
ax.hlines(pfp,0,10000,linestyles="dashdot",linewidth=1,label="fp64 bound",color="black")

ax.set_xlim(s["x"][0],s["x"][-1])
ax.set_ylim(0,pcache+0.1)
ax.yaxis.set_major_locator(MultipleLocator(1)) 
ax.yaxis.set_major_formatter(FormatStrFormatter(' % 1.0f')) 

plt.legend(loc='upper right',ncol=1,fontsize=8)
plt.tight_layout()
fig.savefig("4060ti_subgrids_fp64.png",dpi=600)
plt.close(fig)


#individual kernel
########################################################################################
d="noctua2_7763_kernel_full"
run("python bench.py -c "+d+".json -d -o "+d+".csv -f makes-label,grids,subgrids-0,subgrids-1,mus/it")
df=read(d+".csv")  
D=getdata(df,"grids","mus/it")

fig = plt.figure()
fig, ax = plt.subplots(figsize=(3, 3))
ax.set_xlabel(r'grid size $N$')
ax.set_ylabel(r'grid point update rate [GUP/s]')
plt.text(200,45,'AMD\n7763',fontsize="8")
#ax.set_yscale("log")
#ax.set_xlim(0,120)

s=select(D,{"makes-label":"full"})
ax.plot(s["x"],s["y"],linewidth=1,label="all kernel",color=col[0])
pcache=cachebw["7763_full"]/(buffer_fp32*rw["full"])
pmem=membw["7763_full"]/(buffer_fp32*rw["full"])
ax.hlines(pcache,0,10000,linestyles="dashed",linewidth=1,label="",color=col[0])
ax.hlines(pmem,0,10000,linestyles="dotted",linewidth=1,label="",color=col[0])
ax.set_xlim(s["x"][0],s["x"][-1])

s=select(D,{"makes-label":"calculate_k"})
ax.plot(s["x"],s["y"],linewidth=1,label="only derivative",color=col[1])
pcache=cachebw["7763_full"]/(buffer_fp32*rw["calculate_k"])
pmem=membw["7763_full"]/(buffer_fp32*rw["calculate_k"])
ax.hlines(pcache,0,10000,linestyles="dashed",linewidth=1,label="",color=col[1])
ax.hlines(pmem,0,10000,linestyles="dotted",linewidth=1,label="",color=col[1])


s=select(D,{"makes-label":"intermediate_sum"})
ax.plot(s["x"],s["y"],linewidth=1,label="only two-term-sum",color=col[2])
pcache=cachebw["7763_full"]/(buffer_fp32*rw["intermediate_sum"])
pmem=membw["7763_full"]/(buffer_fp32*rw["intermediate_sum"])
ax.hlines(pcache,0,10000,linestyles="dashed",linewidth=1,label="",color=col[2])
ax.hlines(pmem,0,10000,linestyles="dotted",linewidth=1,label="",color=col[2])


s=select(D,{"makes-label":"final_sum"})
ax.plot(s["x"],s["y"],linewidth=1,label="only five-term-sum",color=col[3])
pcache=cachebw["7763_full"]/(buffer_fp32*rw["final_sum"])
pmem=membw["7763_full"]/(buffer_fp32*rw["final_sum"])
ax.hlines(pcache,0,10000,linestyles="dashed",linewidth=1,label="",color=col[3])
ax.hlines(pmem,0,10000,linestyles="dotted",linewidth=1,label="",color=col[3])
ax.yaxis.set_major_formatter(FormatStrFormatter(' % 1.0f')) 


plt.legend(loc='upper right',ncol=1,fontsize=8)
plt.tight_layout()
fig.savefig("7763_full_kernel_fp32.png",dpi=600)
plt.close(fig)

#individual kernel
########################################################################################
d="noctua2_a100_kernel"
run("python bench.py -c "+d+".json -d -o "+d+".csv -f makes-label,grids,subgrids-0,subgrids-1,mus/it")
df=read(d+".csv")
D=getdata(df,"grids","mus/it")

fig = plt.figure()
fig, ax = plt.subplots(figsize=(3, 3))
ax.set_xlabel(r'grid size $N$')
ax.set_ylabel(r'grid point update rate [GUP/s]')
plt.text(200,90,'NVIDIA\nA100',fontsize="8")
#ax.set_yscale("log")
#ax.set_xlim(0,120)

s=select(D,{"makes-label":"full"})
ax.plot(s["x"],s["y"],linewidth=1,label="all kernel",color=col[0])
pcache=cachebw["a100"]/(buffer_fp32*rw["full"])
pmem=membw["a100"]/(buffer_fp32*rw["full"])
ax.hlines(pcache,0,10000,linestyles="dashed",linewidth=1,label="",color=col[0])
ax.hlines(pmem,0,10000,linestyles="dotted",linewidth=1,label="",color=col[0])
ax.set_xlim(s["x"][0],s["x"][-1])

s=select(D,{"makes-label":"calculate_k"})
ax.plot(s["x"],s["y"],linewidth=1,label="only derivative",color=col[1])
pcache=cachebw["a100"]/(buffer_fp32*rw["calculate_k"])
pmem=membw["a100"]/(buffer_fp32*rw["calculate_k"])
ax.hlines(pcache,0,10000,linestyles="dashed",linewidth=1,label="",color=col[1])
ax.hlines(pmem,0,10000,linestyles="dotted",linewidth=1,label="",color=col[1])


s=select(D,{"makes-label":"intermediate_sum"})
ax.plot(s["x"],s["y"],linewidth=1,label="only two-term-sum",color=col[2])
pcache=cachebw["a100"]/(buffer_fp32*rw["intermediate_sum"])
pmem=membw["a100"]/(buffer_fp32*rw["intermediate_sum"])
ax.hlines(pcache,0,10000,linestyles="dashed",linewidth=1,label="",color=col[2])
ax.hlines(pmem,0,10000,linestyles="dotted",linewidth=1,label="",color=col[2])


s=select(D,{"makes-label":"final_sum"})
ax.plot(s["x"],s["y"],linewidth=1,label="only five-term-sum",color=col[3])
pcache=cachebw["a100"]/(buffer_fp32*rw["final_sum"])
pmem=membw["a100"]/(buffer_fp32*rw["final_sum"])
ax.hlines(pcache,0,10000,linestyles="dashed",linewidth=1,label="",color=col[3])
ax.hlines(pmem,0,10000,linestyles="dotted",linewidth=1,label="",color=col[3])


plt.legend(loc='upper right',ncol=1,fontsize=8)
plt.tight_layout()
fig.savefig("a100_kernel_fp32.png",dpi=600)
plt.close(fig)


#power
###############################################################################################
fig = plt.figure()
fig, ax = plt.subplots(figsize=(5, 3))
ax.set_xlabel(r'grid size $N$')
ax.set_ylabel(r'energy per grid point update [nJ]')

d="noctua2_a100_subgrids"
run("python bench.py -c "+d+".json -d -o "+d+".csv -f makes-label,grids,subgrids-0,subgrids-1,mus/it,gpu_energy_J")
df=read(d+".csv")
D=getdatadiv(df,"grids","mus/it","gpu_energy_J")

s=selectmin(D,{"makes-label":"full_fp32"})
ax.plot(s["x"],s["y"],linewidth=1,label="NVIDIA A100",color=col[0])
s=selectmin(D,{"makes-label":"full_fp64"})
ax.plot(s["x"],s["y"],linewidth=1,label="",linestyle="dashed",color=col[0])

d="4060ti_subgrids"
run("python bench.py -c "+d+".json -d -o "+d+".csv -f makes-label,grids,subgrids-0,subgrids-1,mus/it,gpu_energy_J")
df=read(d+".csv")
D=getdatadiv(df,"grids","mus/it","gpu_energy_J")
s=selectmin(D,{"makes-label":"full_fp32"})
ax.plot(s["x"],s["y"],linewidth=1,label="NVIDIA 4060TI",color=col[1])
s=selectmin(D,{"makes-label":"full_fp64"})
ax.plot(s["x"],s["y"],linewidth=1,label="",linestyle="dashed",color=col[1])

d="noctua2_7763_subgrids_full"
run("python bench.py -c "+d+".json -d -o "+d+".csv -f makes-label,grids,subgrids-0,subgrids-1,mus/it,cpu_energy_J_perf")
df=read(d+".csv")
D=getdatadiv(df,"grids","mus/it","cpu_energy_J_perf")
s=selectmin(D,{"makes-label":"full_fp32"})
ax.plot(s["x"],s["y"],linewidth=1,label="AMD 7763",color=col[2])
s=selectmin(D,{"makes-label":"full_fp64"})
ax.plot(s["x"],s["y"],linewidth=1,label="",linestyle="dashed",color=col[2])

d="5800X3D_subgrids_socket"
run("python bench.py -c "+d+".json -d -o "+d+".csv -f makes-label,grids,subgrids-0,subgrids-1,mus/it,cpu_energy_J_perf")
df=read(d+".csv")
D=getdatadiv(df,"grids","mus/it","cpu_energy_J_perf")
s=selectmin(D,{"makes-label":"full_fp32"})
ax.plot(s["x"],s["y"],linewidth=1,label="AMD 5800X3D",color=col[3])
s=selectmin(D,{"makes-label":"full_fp64"})
ax.plot(s["x"],s["y"],linewidth=1,label="",linestyle="dashed",color=col[3])

ax.set_xlim(s["x"][0],s["x"][-1])
ax.set_ylim(0,650)

plt.legend(loc='upper right',ncol=1,fontsize=8)
plt.tight_layout()
fig.savefig("energy_per_update.png",dpi=600)
plt.close(fig)
