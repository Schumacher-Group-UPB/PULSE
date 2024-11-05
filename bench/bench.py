import json
import os
import argparse
import itertools
import subprocess

def gv(data,c,s):
    i=0
    for f in data:
        if f!="name":
            if f==s:
                break
            else:
                i=i+1
    return c[i]

def runcmd(cmd):
    print("running: ",cmd)
    p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)
    output = p.stdout.read().decode()
    return output

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", help="increase output verbosity",action="store_true")
parser.add_argument("-c","--config", help="json config for benchmask")
args=parser.parse_args()

with open(args.config, 'r') as file:
    data = json.load(file)

#build all combinations
a=[]
for f in data:
    if f!="name":
        a.append(data[f])
print(a)
comb=list(itertools.product(*a))
print("runs=",len(comb))

os.makedirs(os.path.join("runs",data["name"]),exist_ok=True)
rebuild=["commits","envs","makes"]
last={}
for ic in range(len(comb)):
    #build run
    d=os.path.join("runs",data["name"],str(ic))
    os.makedirs(d,exist_ok=True)

    #build code
    bindir="build"
    for i in rebuild:
        print(i,gv(data,comb[ic],i))
        bindir+="_"+gv(data,comb[ic],i)["label"]
    buildscript="#!/bin/bash\n"
    buildscript+="mkdir bins\n"
    buildscript+="cd bins\n"
    buildscript+="git clone "+gv(data,comb[ic],"commits")["repo"]+"\n"
    buildscript+="rm -rf \""+bindir+"\"\n"
    buildscript+="mv PHOENIX \""+bindir+"\"\n"
    buildscript+="cd \""+bindir+"\"\n"
    if gv(data,comb[ic],"commits")["type"]=="branch":
        buildscript+="git branch "+gv(data,comb[ic],"commits")["label"]+" origin/"+gv(data,comb[ic],"commits")["label"]+"\n"
        buildscript+="git checkout "+gv(data,comb[ic],"commits")["label"]+"\n"
    if gv(data,comb[ic],"commits")["type"]=="commit":
        buildscript+="git checkout "+gv(data,comb[ic],"commits")["label"]+"\n"
    env=gv(data,comb[ic],"envs")
    envlabel=env["label"]
    if len(env["modules"])>0:
        buildscript+="module reset\n"
        for m in env["modules"]:
            buildscript+="module load "+m+"\n"
    buildscript+="make clean\n"
    make=gv(data,comb[ic],"makes")
    makecmd=make["cmd"]
    #replace setting from env in make command
    for i in env:
        if isinstance(env[i],str):
            makecmd=makecmd.replace("_"+i+"_",env[i])
    makelabel=make["label"]
    buildscript+=makecmd+"\n"
    with open(os.path.join(d,'build.sh'), 'w') as f:
        f.write(buildscript)

    #build version if necessary
    dorebuild=False
    if len(last)==0:
        for i in rebuild:
            last[i]=gv(data,comb[ic],i)
        dorebuild=True
    else:
        for i in rebuild:
             if last[i]!=gv(data,comb[ic],i):
                dorebuild=True
    if dorebuild:
        for i in rebuild:
            last[i]=gv(data,comb[ic],i)
        print("building ",last)
        runcmd("bash "+os.path.join(d,'build.sh'))

    #build run script
    runscript="#!/bin/bash\n"
    runscript+="cd \""+d+"\"\n"
    runscript+="export OMP_PLACES=cores\n"
    if len(env["modules"])>0:
        runscript+="module reset\n"
        for m in env["modules"]:
            runscript+="module load "+m+"\n"
    runscript+="export OMP_NUM_THREADS="+str(gv(data,comb[ic],"threads"))+"\n"
    runscript+="export OMP_PROC_BIND=true\n"
    runscript+="export OMP_PLACES=cores\n"
    rcmd=""
    if "machinestate" in env:
        runscript+=env["machinestate"]+" > machinestate.json\n"
    runscript+=env["compiler"]+" --version > compiler.version\n"
    if "nvidia-smi" in env:
        runscript+=env["nvidia-smi"]+" --query-gpu=index,timestamp,power.draw,clocks.sm,clocks.mem,clocks.gr --format=csv -lms 100 -i 0 > nvidia.csv & \n"
        runscript+="nvpid=$!\n"
    rcmd=""
    if "perf" in env:
        rcmd=env["perf"]+" stat -e power/energy-pkg/ --per-socket -x \",\" -o perf.out "+rcmd
    if "likwid" in env:
        rcmd=env["likwid"]+" -m -g "+gv(data,comb[ic],"likwid_metrics")+" -C 0-"+str(gv(data,comb[ic],"threads")-1)+" -o likwid.json "+rcmd
    rcmd+="../../../bins/"+bindir+"/main.o -L 200 200 --tmax 1000 --gammaC 0.01 --gc 6E-6 --gr 12E-6 --initRandom 0.01 4242 100 --outEvery 1000 --threads "+str(gv(data,comb[ic],"threads"))+" --path output --N "+str(gv(data,comb[ic],"grids"))+" "+str(gv(data,comb[ic],"grids"))+" --output psi_plus --fftEvery 100000000 --boundary zero zero --subgrids "+str(gv(data,comb[ic],"subgrids")[0])+" "+str(gv(data,comb[ic],"subgrids")[1])+" > out 2> err"
    runscript+=rcmd+"\n"
    if "nvidia-smi" in env:
        runscript+="kill $nvpid\n"
    with open(os.path.join(d,'run.sh'), 'w') as f:
        f.write(runscript)
    runcmd("bash "+os.path.join(d,'run.sh'))

    run={"bindir":os.path.join(os.getcwd(),bindir)}
    #parse results
    #Total Runtime: 10s --> 0.26ms/ps --> 3.84e+03ps/s --> 35.6 mus/it
    with open(os.path.join(d,"out")) as f:
        for l in f.read().splitlines():
            if l.startswith("Total Runtime: "):
                run["wall"]=float(l.split()[2].replace("s",""))
                run["ms/ps"]=float(l.split()[4].replace("ms/ps",""))
                run["ps/s"]=float(l.split()[6].replace("ps/s",""))
                run["mus/it"]=float(l.split()[8].replace("mus/it",""))
                break
    
    #get power usage
    #cpu
    if gv(data,comb[ic],"likwid_metrics")=="ENERGY":
        with open(os.path.join(d,"likwid.json")) as f:
            likwid_data = json.load(f)
        Pcpu=likwid_data["Energy"]["Energy"]["Metric STAT"]["Power PKG [W] STAT"]
   
    #gpu
    Pgpu=-1
    if "nvidia-smi" in env:
        with open(os.path.join(d,"nvidia.csv")) as f:
            for l in f.read().splitlines():
                Pgpu=flaot(l.split(",")[2].split(" ")[1])
           
    if Pgpu!=-1:
        run["gpu_power_W"]=Pgpu
    if Pcpu!=-1:
        run["cpu_power_W"]=Pcpu



    #write json description file
    for f in data:
        if f!="name":
            run[f]=gv(data,comb[ic],f)
    with open(os.path.join(d,'run.json'), 'w') as f:
            json.dump(run, f)








