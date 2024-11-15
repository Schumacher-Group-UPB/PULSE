import json
import os
import argparse
import itertools
import subprocess

def gv(data,c,s):
    i=0
    for f in data:
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
parser.add_argument("-c","--config", help="json config for benchmask")
parser.add_argument("-o","--output", help="output csv file",default="result.csv")
parser.add_argument("-f","--fields", help="fields to include in output",default="grids,subgrids,threads")
parser.add_argument("-d","--dryrun", help="only write output file from existing data, don't build and run benchmarks",action='store_true')
args=parser.parse_args()

with open(args.config, 'r') as file:
    data = json.load(file)

name=args.config.replace(".json","")

#build all combinations
a=[]
for f in data:
    a.append(data[f])
comb=list(itertools.product(*a))
print("runs=",len(comb))

if not args.dryrun:
    os.makedirs(os.path.join("runs",name),exist_ok=True)
    rebuild=["commits","envs","makes"]
    last={}
    for ic in range(len(comb)):
        #build run
        d=os.path.join("runs",name,str(ic))
        os.makedirs(d,exist_ok=True)

        #build code
        bindir="build"
        for i in rebuild:
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
            rcmd=env["perf"]+" stat -e power/energy-pkg/ --per-socket -x \";\" -o perf.out "+rcmd
        if "likwid" in env:
            rcmd=env["likwid"]+" -m -g "+gv(data,comb[ic],"likwid_metrics")+" -C 0-"+str(gv(data,comb[ic],"threads")-1)+" -o likwid.json "+rcmd
        rcmd+="../../../bins/"+bindir+"/main.o -L 200 200 --tmax 1000 --gammaC 0.01 --gc 6E-6 --gr 12E-6 --initRandom 0.01 4242 100 --outEvery 1000 --threads "+str(gv(data,comb[ic],"threads"))+" --path output --N "+str(gv(data,comb[ic],"grids"))+" "+str(gv(data,comb[ic],"grids"))+" --output psi_plus --fftEvery 100000000 --boundary zero zero --subgrids "+str(gv(data,comb[ic],"subgrids")[0])+" "+str(gv(data,comb[ic],"subgrids")[1])+" > out 2> err"
        runscript+=rcmd+"\n"
        if "nvidia-smi" in env:
            runscript+="kill $nvpid\n"
        with open(os.path.join(d,'run.sh'), 'w') as f:
            f.write(runscript)
        runcmd("bash "+os.path.join(d,'run.sh'))

        #analyse benchmark and write result file
        
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
        
        #map likwid jsons fileds to result fields
        lmap={"ENERGY":["Energy PKG [J] STAT","Sum"],"DATA":["Load to store ratio STAT","avg"],"FLOPS_SP":["SP [MFLOP/s] STAT","Sum"],"CACHE":["data cache miss ratio STAT","Avg"],"L2":["L2 bandwidth [MBytes/s] STAT","Sum"],"L2CACHE":["L2 miss ratio STAT","Avg"],"L3":["L3 bandwidth [MBytes/s] STAT","Sum"],"MEM1":["Memory bandwidth (channels 0-3) [MBytes/s] STAT","sum"],"MEM2":["Memory bandwidth (channels 4-7) [MBytes/s] STAT","Sum"]}
        #get power usage
        #cpu
        Ecpu=0
        if "perf" in env:
            with open(os.path.join(d,"perf.out")) as f:
                for l in f.read().splitlines():
                    if l.startswith("S0"):
                        Ecpu=float(l.split(";")[2])
                        break
        Ecpu2=0
        if "likwid" in env:
            if gv(data,comb[ic],"likwid_metrics")=="ENERGY":
                with open(os.path.join(d,"likwid.json")) as f:
                    likwid_data = json.load(f)
                Ecpu2=likwid_data["ENERGY"]["ENERGY"]["Metric STAT"][lmap["ENERGY"][0]][lmap["ENERGY"][1]]
       
        #gpu
        Egpu=0
        if "nvidia-smi" in env:
            with open(os.path.join(d,"nvidia.csv")) as f:
                for l in f.read().splitlines():
                    if not l.find("index,")>=0:
                        Egpu+=float(l.split(",")[2].split(" ")[1])*0.1
               
        run["gpu_energy_J"]=Egpu
        run["cpu_energy_J_perf"]=Ecpu
        run["cpu_energy_J_likwid"]=Ecpu2

        Mlikwid=0
        if "likwid" in env:
            try:
                if gv(data,comb[ic],"likwid_metrics")!="":
                    with open(os.path.join(d,"likwid.json")) as f:
                        likwid_data = json.load(f)
                    Mlikwid=likwid_data[gv(data,comb[ic],"likwid_metrics")][gv(data,comb[ic],"likwid_metrics")]["Metric STAT"][lmap[gv(data,comb[ic],"likwid_metrics")][0]][lmap[gv(data,comb[ic],"likwid_metrics")][1]]
            except:
                pass
        run["likwid_measurement"]=Mlikwid


        #write json description file
        for f in data:
            run[f]=gv(data,comb[ic],f)
        with open(os.path.join(d,'run.json'), 'w') as f:
                json.dump(run, f)

#build csv output file
print("fields for output:",args.fields)
res="#"
for f in args.fields.split(","):
    if not f.find("=")>=0:
        res+=f+";"
res+="\n"

for ic in range(len(comb)):
    d=os.path.join("runs",name,str(ic))
    if not os.path.isfile(os.path.join(d,"run.json")):
        continue
    with open(os.path.join(d,"run.json"), 'r') as file:
        data = json.load(file)
    add=True
    for f in args.fields.split(","):
        #check if is filter
        if f.find("=")>=0:
            g=f.split("=")[0]
            if len(g.split("-"))==1:
                v=f.split("=")[1]
                if str(data[g]).strip()!=str(v).strip():
                    add=False
            else:
                g0=g.split("-")[0]
                g1=g.split("-")[1]
                v=f.split("=")[1]
                if str(data[g0][g1]).strip()!=str(v).strip():
                    add=False
    #build line
    if add:
        for f in args.fields.split(","):
            if not f.find("=")>=0:
                try:
                    if f.find("-")>=0:
                        f1=f.split("-")[0]
                        f2=f.split("-")[1]
                        if isinstance(data[f1],list):
                            s=str(data[f1][int(f2)])
                        else:
                            s=str(data[f1][f2])
                        res+=s+";"
                    else:
                        s=str(data[f])
                        res+=s+";"
                except:
                    res+=";"
                    pass
        res+="\n"

with open(args.output, 'w') as f:
    f.write(res)
    











