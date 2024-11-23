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

def lmax(s):
    m=0
    for v in s:
        m=max(v,m)
    return m

#plt.rcParams['text.usetex'] = True

col=[]
for i in mcolors.TABLEAU_COLORS:
    col.append(i)

#bandwidths in MB/s
cachebw={"7763_ccd":517.22370,"a100":5400.000,"7763_full":2600.000,"5800X3D":580,"4060ti":1600,"4090":5300,"h100":8000}
membw={"7763_ccd":38.60948,"a100":1400.000,"7763_full":155.000,"5800X3D":31,"4060ti":275,"4090":950,"h100":3200}
fp={"4060ti":0.3,"5800X3D":0.5,"4090":1.3}

buffer_fp32=8
buffer_fp64=16
rw={"full":23+8,"calculate_k":4*(3+1),"intermediate_sum":3*(2+1),"final_sum":1*(5+1)}

#cache plot
########################################################################################
fig = plt.figure()
fig, ax = plt.subplots(figsize=(5, 3))
#ax.set_xlabel(r'grid size $N$')
ax.set_xlabel(r'number of grid points per spatial dimension $N$')
ax.set_ylabel(r'LLC cache size [MiB]')
#ax.set_yscale("log")
#ax.set_xlim(0,120)

y=[]
y2=[]
x=list(range(0,3000))
for X in x:
    y.append(X*X*buffer_fp32*6/1024/1024)
    y2.append(X*X*buffer_fp64*6/1024/1024)
ax.plot(x,y,linewidth=1,label="active data size in fp32",color="black")
ax.plot(x,y2,linewidth=1,label="active data size in fp64",linestyle="dashed",color="black")

ax.hlines(32,0,10000,linestyles="dashed",linewidth=1,label="LLC size NVIDIA 4060 TI",color=col[0])
ax.hlines(72,0,10000,linestyles="dashed",linewidth=1,label="LLC size NVIDIA 4090",color=col[1])
ax.hlines(50,0,10000,linestyles="dashed",linewidth=1,label="LLC size NVIDIA H100",color=col[2])
ax.hlines(96,0,10000,linestyles="dashed",linewidth=1,label="LLC size AMD 5800X3D",color=col[3])
ax.hlines(256,0,10000,linestyles="dashed",linewidth=1,label="LLC size AMD 7763",color=col[4])

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
plt.text(50,180,'cache bandwidth bound',fontsize="8")
plt.text(1500,53,'memory bandwidth bound',fontsize="8")
#ax.set_yscale('log')
plt.legend(loc='upper left',ncol=1,fontsize="7")
plt.tight_layout()
fig.savefig("cache_sizes.png",dpi=600,bbox_inches = 'tight',pad_inches = 0)
plt.close(fig)


meta=[]
meta.append({"i":0,"j":0,"xlim":[64,4096],"yoffset":0.1,"dev":"4060ti","text":'NVIDIA\n4060 TI\nfp32',"text_x":950,"text_y":5.0,"d":"4060ti_subgrids","fp":"fp32","sg":[{"x":1,"y":1,"l":"no subgrids"},{"x":2,"y":1,"l":"2x1"},{"x":4,"y":1,"l":"4x1"},{"x":8,"y":1,"l":"8x1"}],"xticks":2,"xspace":1})
meta.append({"i":0,"j":1,"xlim":[64,4096],"yoffset":0.05,"dev":"4060ti","text":'NVIDIA\n4060 TI\nfp64',"text_x":400,"text_y":2.5,"d":"4060ti_subgrids","fp":"fp64","sg":[{"x":1,"y":1,"l":"no subgrids"},{"x":2,"y":1,"l":"2x1"},{"x":4,"y":1,"l":"4x1"},{"x":8,"y":1,"l":"8x1"}],"xticks":1,"xspace":1})

meta.append({"i":1,"j":0,"xlim":[64,4096],"yoffset":0.2,"dev":"4090","text":'NVIDIA\n4090\nfp32',"text_x":400,"text_y":17.0,"d":"4090_subgrids","fp":"fp32","sg":[{"x":1,"y":1,"l":"no subgrids"},{"x":2,"y":1,"l":"2x1"},{"x":4,"y":1,"l":"4x1"},{"x":8,"y":1,"l":"8x1"}],"xticks":5,"xspace":0})
meta.append({"i":1,"j":1,"xlim":[64,4096],"yoffset":0.1,"dev":"4090","text":'NVIDIA\n4090\nfp64',"text_x":400,"text_y":8.5,"d":"4090_subgrids","fp":"fp64","sg":[{"x":1,"y":1,"l":"no subgrids"},{"x":2,"y":1,"l":"2x1"},{"x":4,"y":1,"l":"4x1"},{"x":8,"y":1,"l":"8x1"}],"xticks":2,"xspace":0})

#meta.append({"i":2,"j":0,"xlim":[64,4096],"yoffset":0.35,"dev":"a100","text":'NVIDIA\nA100\nfp32',"text_x":400,"text_y":17.0,"d":"noctua2_a100_subgrids","fp":"fp32","sg":[{"x":1,"y":1,"l":"no subgrids"},{"x":2,"y":1,"l":"2x1"},{"x":4,"y":1,"l":"4x1"},{"x":8,"y":1,"l":"8x1"}],"xticks":5,"xspace":0})
#meta.append({"i":2,"j":1,"xlim":[64,4096],"yoffset":0.15,"dev":"a100","text":'NVIDIA\nA100\nfp64',"text_x":400,"text_y":8.5,"d":"noctua2_a100_subgrids","fp":"fp64","sg":[{"x":1,"y":1,"l":"no subgrids"},{"x":2,"y":1,"l":"2x1"},{"x":4,"y":1,"l":"4x1"},{"x":8,"y":1,"l":"8x1"}],"xticks":2,"xspace":0})

meta.append({"i":2,"j":0,"xlim":[64,4096],"yoffset":0.35,"dev":"h100","text":'NVIDIA\nH100\nfp32',"text_x":400,"text_y":25.0,"d":"h100_subgrids","fp":"fp32","sg":[{"x":1,"y":1,"l":"no subgrids"},{"x":2,"y":1,"l":"2x1"},{"x":4,"y":1,"l":"4x1"},{"x":8,"y":1,"l":"8x1"}],"xticks":5,"xspace":0})
meta.append({"i":2,"j":1,"xlim":[64,4096],"yoffset":0.15,"dev":"h100","text":'NVIDIA\nH100\nfp64',"text_x":400,"text_y":12.5,"d":"h100_subgrids","fp":"fp64","sg":[{"x":1,"y":1,"l":"no subgrids"},{"x":2,"y":1,"l":"2x1"},{"x":4,"y":1,"l":"4x1"},{"x":8,"y":1,"l":"8x1"}],"xticks":2,"xspace":0})

meta.append({"i":3,"j":0,"xlim":[64,4096],"yoffset":0.025,"dev":"5800X3D","text":'AMD\n5800X3D\nfp32',"text_x":400,"text_y":1.8,"d":"5800X3D_subgrids_socket","fp":"fp32","sg":[{"x":2,"y":4,"l":"2x4"},{"x":4,"y":4,"l":"4x4"},{"x":8,"y":4,"l":"8x4"},{"x":8,"y":8,"l":"8x8"},{"x":16,"y":8,"l":"16x8"}],"xticks":1,"xspace":1})
meta.append({"i":3,"j":1,"xlim":[64,4096],"yoffset":0.0125,"dev":"5800X3D","text":'AMD\n5800X3D\nfp64',"text_x":400,"text_y":0.9,"d":"5800X3D_subgrids_socket","fp":"fp64","sg":[{"x":2,"y":4,"l":"2x4"},{"x":4,"y":4,"l":"4x4"},{"x":8,"y":4,"l":"8x4"},{"x":8,"y":8,"l":"8x8"},{"x":16,"y":8,"l":"16x8"}],"xticks":0.5,"xspace":1})

meta.append({"i":4,"j":0,"xlim":[64,4096],"yoffset":0.025,"dev":"7763_ccd","text":'AMD 7763\none CCD\nfp32',"text_x":400,"text_y":1.6,"d":"noctua2_7763_subgrids_ccd","fp":"fp32","sg":[{"x":2,"y":4,"l":"2x4"},{"x":4,"y":4,"l":"4x4"},{"x":8,"y":4,"l":"8x4"},{"x":8,"y":8,"l":"8x8"},{"x":16,"y":8,"l":"16x8"}],"xticks":1,"xspace":1})
meta.append({"i":4,"j":1,"xlim":[64,4096],"yoffset":0.0125,"dev":"7763_ccd","text":'AMD 7763\none CCD\nfp64',"text_x":400,"text_y":0.8,"d":"noctua2_7763_subgrids_ccd","fp":"fp64","sg":[{"x":2,"y":4,"l":"2x4"},{"x":4,"y":4,"l":"4x4"},{"x":8,"y":4,"l":"8x4"},{"x":8,"y":8,"l":"8x8"},{"x":16,"y":8,"l":"16x8"}],"xticks":0.5,"xspace":0})

meta.append({"i":5,"j":0,"xlim":[64,4096],"yoffset":0.1,"dev":"7763_full","text":'AMD 7763\nall CCDs\nfp32',"text_x":400,"text_y":8,"d":"noctua2_7763_subgrids_full","fp":"fp32","sg":[{"x":8,"y":8,"l":"8x8"},{"x":16,"y":8,"l":"16x8"},{"x":16,"y":16,"l":"16x16"},{"x":16,"y":32,"l":"16x32"},{"x":32,"y":32,"l":"32x32"}],"xticks":2,"xspace":0})
meta.append({"i":5,"j":1,"xlim":[64,4096],"yoffset":0.05,"dev":"7763_full","text":'AMD 7763\nall CCDs\nfp64',"text_x":400,"text_y":4,"d":"noctua2_7763_subgrids_full","fp":"fp64","sg":[{"x":8,"y":8,"l":"8x8"},{"x":16,"y":8,"l":"16x8"},{"x":16,"y":16,"l":"16x16"},{"x":16,"y":32,"l":"16x32"},{"x":32,"y":32,"l":"32x32"}],"xticks":1,"xspace":1})


#subgrid dependency: all
########################################################################################
#subplots for alignment
##########################
fig, axs = plt.subplots(6, 2, sharex=True, sharey=False, layout="constrained",figsize=(9, 13))
for m in meta:
    try:
        i=m["i"]
        j=m["j"]
        d=m["d"]
        run("python bench.py -c "+d+".json -d -o "+d+".csv -f makes-label,makes-label=full_"+m["fp"]+",grids,subgrids-0,subgrids-1,mus/it")

        df=read(d+".csv")  
        D=getdata(df,"grids","mus/it")

        axs[i,j].text(m["text_x"],m["text_y"],m["text"],fontsize="8")
        axs[i,j].set_xlim(m["xlim"][0],m["xlim"][1])

        pm=0
        ic=0
        for sg in m["sg"]:
            s=select(D,{"makes-label":"full_"+m["fp"],"subgrids-0":sg["x"],"subgrids-1":sg["y"]})
            axs[i,j].plot(s["x"],s["y"],linewidth=1,label=sg["l"],color=col[ic])
            pm=max(pm,lmax(s["y"]))
            ic=ic+1

        #cache-bw
        if m["fp"]=="fp32":
            pcache=cachebw[m["dev"]]/(buffer_fp32*rw["full"])
            pmem=membw[m["dev"]]/(buffer_fp32*rw["full"])
        else:
            pcache=cachebw[m["dev"]]/(buffer_fp64*rw["full"])
            pmem=membw[m["dev"]]/(buffer_fp64*rw["full"])

        print("max",d,pm,pcache,pmem,pm/pcache,pm/pmem)
        axs[i,j].yaxis.set_major_locator(MultipleLocator(m["xticks"])) 
        if m["xticks"]>=1:
            if m["xspace"]==0:
                axs[i,j].yaxis.set_major_formatter(FormatStrFormatter('% 1.0f')) 
            if m["xspace"]==1:
                axs[i,j].yaxis.set_major_formatter(FormatStrFormatter(' % 1.0f')) 
            if m["xspace"]==2:
                axs[i,j].yaxis.set_major_formatter(FormatStrFormatter('  % 1.0f')) 


        axs[i,j].hlines(pcache,0,10000,linestyles="dashed",linewidth=1,label="L2 cache bound",color="black")
        axs[i,j].hlines(pmem,0,10000,linestyles="dotted",linewidth=1,label="mem bound",color="black")
        if m["fp"]=="fp64":
            if m["dev"] in  fp:
                pfp=fp[m["dev"]]/(buffer_fp64*rw["full"]*0.333)*1000
                axs[i,j].hlines(pfp,0,10000,linestyles="dashdot",linewidth=1,label="fp64 bound",color="black")
        axs[i,j].set_ylim(0,pcache+m["yoffset"])
        #axs[i,j].yaxis.set_major_locator(MultipleLocator(2)) 
        #axs[i,j].yaxis.set_major_formatter(FormatStrFormatter(' % 1.0f')) 
        axs[i,j].legend(loc='upper right',ncol=1,fontsize=8)
    except:
        pass

########################################################################################
########################################################################################

axs[0,0].set_ylabel(r'point update rate [GUP/s]')
axs[1,0].set_ylabel(r'point update rate [GUP/s]')
axs[2,0].set_ylabel(r'point update rate [GUP/s]')
axs[3,0].set_ylabel(r'point update rate [GUP/s]')
axs[4,0].set_ylabel(r'point update rate [GUP/s]')
axs[5,0].set_ylabel(r'point update rate [GUP/s]')
#axs[5,0].set_xlabel(r'grid size $N$')
#axs[5,1].set_xlabel(r'grid size $N$')
#axs[5,0].set_xlabel(r'number of grid points per spatial dimension $N$')
#axs[5,1].set_xlabel(r'grid size $N$')
fig.text(0.5, -0.01, r'number of grid points per spatial dimension $N$', ha='center')

fig.savefig("subgrids.png",dpi=600,bbox_inches = 'tight',pad_inches = 0)
plt.close(fig)

#subgrid dependency: all
########################################################################################
#subplots for alignment
##########################
fig, axs = plt.subplots(6, 2, sharex=True, sharey=False, layout="constrained",figsize=(6, 14))
for m in meta:
    try:
        i=m["i"]
        j=m["j"]
        d=m["d"]
        run("python bench.py -c "+d+".json -d -o "+d+".csv -f makes-label,makes-label=full_"+m["fp"]+",grids,subgrids-0,subgrids-1,mus/it")

        df=read(d+".csv")  
        D=getdata(df,"grids","mus/it")

        axs[i,j].text(m["text_x"],m["text_y"],m["text"],fontsize="8")
        axs[i,j].set_xlim(m["xlim"][0],m["xlim"][1])

        pm=0
        ic=0
        for sg in m["sg"]:
            s=select(D,{"makes-label":"full_"+m["fp"],"subgrids-0":sg["x"],"subgrids-1":sg["y"]})
            axs[i,j].plot(s["x"],s["y"],linewidth=1,label=sg["l"],color=col[ic])
            pm=max(pm,lmax(s["y"]))
            ic=ic+1

        #cache-bw
        if m["fp"]=="fp32":
            pcache=cachebw[m["dev"]]/(buffer_fp32*rw["full"])
            pmem=membw[m["dev"]]/(buffer_fp32*rw["full"])
        else:
            pcache=cachebw[m["dev"]]/(buffer_fp64*rw["full"])
            pmem=membw[m["dev"]]/(buffer_fp64*rw["full"])

        print("max",d,pm,pcache,pmem,pm/pcache,pm/pmem)
        axs[i,j].yaxis.set_major_locator(MultipleLocator(m["xticks"])) 
        if m["xticks"]>=1:
            if m["xspace"]==0:
                axs[i,j].yaxis.set_major_formatter(FormatStrFormatter('% 1.0f')) 
            if m["xspace"]==1:
                axs[i,j].yaxis.set_major_formatter(FormatStrFormatter(' % 1.0f')) 
            if m["xspace"]==2:
                axs[i,j].yaxis.set_major_formatter(FormatStrFormatter('  % 1.0f')) 


        axs[i,j].hlines(pcache,0,10000,linestyles="dashed",linewidth=1,label="L2 cache bound",color="black")
        axs[i,j].hlines(pmem,0,10000,linestyles="dotted",linewidth=1,label="mem bound",color="black")
        if m["fp"]=="fp64":
            if m["dev"] in  fp:
                pfp=fp[m["dev"]]/(buffer_fp64*rw["full"]*0.333)*1000
                axs[i,j].hlines(pfp,0,10000,linestyles="dashdot",linewidth=1,label="fp64 bound",color="black")
        axs[i,j].set_ylim(0,pcache+m["yoffset"])
        #axs[i,j].yaxis.set_major_locator(MultipleLocator(2)) 
        #axs[i,j].yaxis.set_major_formatter(FormatStrFormatter(' % 1.0f')) 
        axs[i,j].legend(loc='upper right',ncol=1,fontsize=8)
    except:
        pass

########################################################################################
########################################################################################

axs[0,0].set_ylabel(r'point update rate [GUP/s]')
axs[1,0].set_ylabel(r'point update rate [GUP/s]')
axs[2,0].set_ylabel(r'point update rate [GUP/s]')
axs[3,0].set_ylabel(r'point update rate [GUP/s]')
axs[4,0].set_ylabel(r'point update rate [GUP/s]')
axs[5,0].set_ylabel(r'point update rate [GUP/s]')
#axs[5,0].set_xlabel(r'grid size $N$')
#axs[5,1].set_xlabel(r'grid size $N$')
#axs[5,0].set_xlabel(r'number of grid points per spatial dimension $N$')
#axs[5,1].set_xlabel(r'grid size $N$')
fig.text(0.5, -0.01, r'number of grid points per spatial dimension $N$', ha='center')

fig.savefig("subgrids_tall.png",dpi=600,bbox_inches = 'tight',pad_inches = 0)
plt.close(fig)


#individual kernel
########################################################################################
fig, axs = plt.subplots(1, 2, sharex=True, sharey=False, layout="constrained",figsize=(6, 3))

axs[1].set_xlim(64,4096)

d="noctua2_7763_kernel_full"
run("python bench.py -c "+d+".json -d -o "+d+".csv -f makes-label,grids,subgrids-0,subgrids-1,mus/it")
df=read(d+".csv")  
D=getdata(df,"grids","mus/it")

#axs[0,0]set_xlabel(r'grid size $N$')
#axs[0].set_xlabel(r'number of grid points per spatial dimension $N$')
axs[1].text(200,45,'AMD\n7763',fontsize="8")
#axs[0,0]set_xlim(0,120)

s=select(D,{"makes-label":"full"})
axs[1].plot(s["x"],s["y"],linewidth=1,label="all kernel",color=col[0])
pcache=cachebw["7763_full"]/(buffer_fp32*rw["full"])
pmem=membw["7763_full"]/(buffer_fp32*rw["full"])
axs[1].hlines(pcache,0,10000,linestyles="dashed",linewidth=1,label="",color=col[0])
axs[1].hlines(pmem,0,10000,linestyles="dotted",linewidth=1,label="",color=col[0])


s=select(D,{"makes-label":"calculate_k"})
axs[1].plot(s["x"],s["y"],linewidth=1,label="only derivative",color=col[1])
pcache=cachebw["7763_full"]/(buffer_fp32*rw["calculate_k"])
pmem=membw["7763_full"]/(buffer_fp32*rw["calculate_k"])
axs[1].hlines(pcache,0,10000,linestyles="dashed",linewidth=1,label="",color=col[1])
axs[1].hlines(pmem,0,10000,linestyles="dotted",linewidth=1,label="",color=col[1])


s=select(D,{"makes-label":"intermediate_sum"})
axs[1].plot(s["x"],s["y"],linewidth=1,label="only two-term-sum",color=col[2])
pcache=cachebw["7763_full"]/(buffer_fp32*rw["intermediate_sum"])
pmem=membw["7763_full"]/(buffer_fp32*rw["intermediate_sum"])
axs[1].hlines(pcache,0,10000,linestyles="dashed",linewidth=1,label="",color=col[2])
axs[1].hlines(pmem,0,10000,linestyles="dotted",linewidth=1,label="",color=col[2])


s=select(D,{"makes-label":"final_sum"})
axs[1].plot(s["x"],s["y"],linewidth=1,label="only five-term-sum",color=col[3])
pcache=cachebw["7763_full"]/(buffer_fp32*rw["final_sum"])
pmem=membw["7763_full"]/(buffer_fp32*rw["final_sum"])
axs[1].hlines(pcache,0,10000,linestyles="dashed",linewidth=1,label="",color=col[3])
axs[1].hlines(pmem,0,10000,linestyles="dotted",linewidth=1,label="",color=col[3])

axs[1].legend(loc='upper right',ncol=1,fontsize=8)

#s=select(D,{"makes-label":"parallelization"})
#axs[0,0]plot(s["x"],s["y"],linewidth=1,label="parallelization",color=col[4])
#ax.set_yscale("log")
#ax.set_ylim(0.5,300)


#individual kernel
########################################################################################
d="h100_kernel"
run("python bench.py -c "+d+".json -d -o "+d+".csv -f makes-label,grids,subgrids-0,subgrids-1,mus/it")
df=read(d+".csv")
D=getdata(df,"grids","mus/it")

#axs[0,1]set_xlabel(r'grid size $N$')
#axs[1].set_xlabel(r'number of grid points per spatial dimension $N$')
#axs[1].set_ylabel(r'grid point update rate [GUP/s]')
axs[0].text(200,135,'NVIDIA\nH100',fontsize="8")
#axs[0,1]set_yscale("log")
#axs[0,1]set_xlim(0,120)

try:
    s=select(D,{"makes-label":"full"})
    axs[0].plot(s["x"],s["y"],linewidth=1,label="all kernel",color=col[0])
    pcache=cachebw["h100"]/(buffer_fp32*rw["full"])
    pmem=membw["h100"]/(buffer_fp32*rw["full"])
    axs[0].hlines(pcache,0,10000,linestyles="dashed",linewidth=1,label="",color=col[0])
    axs[0].hlines(pmem,0,10000,linestyles="dotted",linewidth=1,label="",color=col[0])
    axs[0].set_xlim(s["x"][0],s["x"][-1])
except: 
    pass

try:
    s=select(D,{"makes-label":"calculate_k"})
    axs[0].plot(s["x"],s["y"],linewidth=1,label="only derivative",color=col[1])
    pcache=cachebw["h100"]/(buffer_fp32*rw["calculate_k"])
    pmem=membw["h100"]/(buffer_fp32*rw["calculate_k"])
    axs[0].hlines(pcache,0,10000,linestyles="dashed",linewidth=1,label="",color=col[1])
    axs[0].hlines(pmem,0,10000,linestyles="dotted",linewidth=1,label="",color=col[1])
except: 
    pass


try:
    s=select(D,{"makes-label":"intermediate_sum"})
    axs[0].plot(s["x"],s["y"],linewidth=1,label="only two-term-sum",color=col[2])
    pcache=cachebw["h100"]/(buffer_fp32*rw["intermediate_sum"])
    pmem=membw["h100"]/(buffer_fp32*rw["intermediate_sum"])
    axs[0].hlines(pcache,0,10000,linestyles="dashed",linewidth=1,label="",color=col[2])
    axs[0].hlines(pmem,0,10000,linestyles="dotted",linewidth=1,label="",color=col[2])
except: 
    pass


try:
    s=select(D,{"makes-label":"final_sum"})
    axs[0].plot(s["x"],s["y"],linewidth=1,label="only five-term-sum",color=col[3])
    pcache=cachebw["h100"]/(buffer_fp32*rw["final_sum"])
    pmem=membw["h100"]/(buffer_fp32*rw["final_sum"])
    axs[0].hlines(pcache,0,10000,linestyles="dashed",linewidth=1,label="",color=col[3])
    axs[0].hlines(pmem,0,10000,linestyles="dotted",linewidth=1,label="",color=col[3])
except: 
    pass

#s=select(D,{"makes-label":"parallelization"})
#axs[0,1]plot(s["x"],s["y"],linewidth=1,label="parallelization",color=col[4])
#axs[0,1]set_yscale("log")
#axs[0].set_xlim(s["x"][0],s["x"][-1])
#ax.set_ylim(3,400)
axs[0].legend(loc='upper right',ncol=1,fontsize=8)

axs[0].set_ylabel(r'grid point update rate [GUP/s]')
plt.tight_layout()
fig.text(0.5, -0.01, r'number of grid points per spatial dimension $N$', ha='center')
fig.savefig("kernel_fp32.png",dpi=600,bbox_inches = 'tight',pad_inches = 0)
plt.close(fig)

#power
###############################################################################################
fig = plt.figure()
fig, ax = plt.subplots(figsize=(5, 3))
#ax.set_xlabel(r'grid size $N$')
ax.set_xlabel(r'number of grid points per spatial dimension $N$')
ax.set_ylabel(r'energy per grid point update [nJ]')

popt32=1000000
popt64=1000000

d="4060ti_subgrids"
run("python bench.py -c "+d+".json -d -o "+d+".csv -f makes-label,grids,subgrids-0,subgrids-1,mus/it,gpu_energy_J")
df=read(d+".csv")
D=getdatadiv(df,"grids","mus/it","gpu_energy_J")
s=selectmin(D,{"makes-label":"full_fp32"})
print(d,"fp32",min(s["y"]))
ax.plot(s["x"],s["y"],linewidth=1,label="NVIDIA 4060TI",color=col[0])
s=selectmin(D,{"makes-label":"full_fp64"})
print(d,"fp64",min(s["y"]))
ax.plot(s["x"],s["y"],linewidth=1,label="",linestyle="dashed",color=col[0])

d="4090_subgrids"
run("python bench.py -c "+d+".json -d -o "+d+".csv -f makes-label,grids,subgrids-0,subgrids-1,mus/it,gpu_energy_J")
df=read(d+".csv")
D=getdatadiv(df,"grids","mus/it","gpu_energy_J")
s=selectmin(D,{"makes-label":"full_fp32"})
print(d,"fp32",min(s["y"]))
ax.plot(s["x"],s["y"],linewidth=1,label="NVIDIA 4090",color=col[1])
s=selectmin(D,{"makes-label":"full_fp64"})
print(d,"fp64",min(s["y"]))
ax.plot(s["x"],s["y"],linewidth=1,label="",linestyle="dashed",color=col[1])

#d="noctua2_a100_subgrids"
d="h100_subgrids"
run("python bench.py -c "+d+".json -d -o "+d+".csv -f makes-label,grids,subgrids-0,subgrids-1,mus/it,gpu_energy_J")
df=read(d+".csv")
D=getdatadiv(df,"grids","mus/it","gpu_energy_J")
s=selectmin(D,{"makes-label":"full_fp32"})
print(d,"fp32",min(s["y"]))
ax.plot(s["x"],s["y"],linewidth=1,label="NVIDIA H100",color=col[2])
try:
    s=selectmin(D,{"makes-label":"full_fp64"})
    print(d,"fp64",min(s["y"]))
    ax.plot(s["x"],s["y"],linewidth=1,label="",linestyle="dashed",color=col[2])
except:
    pass


d="5800X3D_subgrids_socket"
run("python bench.py -c "+d+".json -d -o "+d+".csv -f makes-label,grids,subgrids-0,subgrids-1,mus/it,cpu_energy_J_perf")
df=read(d+".csv")
D=getdatadiv(df,"grids","mus/it","cpu_energy_J_perf")
s=selectmin(D,{"makes-label":"full_fp32"})
print(d,"fp32",min(s["y"]))
ax.plot(s["x"],s["y"],linewidth=1,label="AMD 5800X3D",color=col[3])
s=selectmin(D,{"makes-label":"full_fp64"})
print(d,"fp64",min(s["y"]))
ax.plot(s["x"],s["y"],linewidth=1,label="",linestyle="dashed",color=col[3])

d="noctua2_7763_subgrids_full"
run("python bench.py -c "+d+".json -d -o "+d+".csv -f makes-label,grids,subgrids-0,subgrids-1,mus/it,cpu_energy_J_perf")
df=read(d+".csv")
D=getdatadiv(df,"grids","mus/it","cpu_energy_J_perf")
s=selectmin(D,{"makes-label":"full_fp32"})
print(d,"fp32",min(s["y"]))
ax.plot(s["x"],s["y"],linewidth=1,label="AMD 7763",color=col[4])
s=selectmin(D,{"makes-label":"full_fp64"})
print(d,"fp64",min(s["y"]))
ax.plot(s["x"],s["y"],linewidth=1,label="",linestyle="dashed",color=col[4])

ax.set_xlim(s["x"][0],s["x"][-1])
ax.set_ylim(0,650)

plt.legend(loc='upper right',ncol=1,fontsize=8)
plt.tight_layout()
fig.savefig("energy_per_update.png",dpi=600)#,bbox_inches = 'tight',pad_inches = 0)
plt.close(fig)
