#Fucntions for SA-GPR model creation and testing using TENSOAP

import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error
import subprocess
import pandas as pd
from pandas import DataFrame
import os
import time
import pickle
import ase
from ase import io

def keys_grabber(category):
    keys = list(pickle.load(open('Data/' + category + '/uid_index.pkl','rb')).keys())
    for i in keys:
        if not os.path.exists('Data/'+category+'/' + str(i) + '.magres'):
            keys.remove(i)
    return keys


def cart_pred_puller(to_read = 'prediction_cartesian.txt'):
    #Pulls out tensors from predict file
    file  = open(to_read, 'r')
    text = file.readlines()
    target = []
    predicted = []

    for i in text:
        i = i.split()
        target.append(np.array(i[0:9]))
        predicted.append(np.array(i[9:]))

    target = (np.array(target)).astype(np.float)
    predicted = (np.array(predicted)).astype(np.float)
    file.close()
    return target, predicted

def l0_pred_puller(to_read):
    file  = open(to_read, 'r')
    text = file.readlines()
    target = []
    predicted = []
    
    for i in text:
        i = i.split()        
        target.append(i[0])
        predicted.append(i[1])
        
    target = (np.array(target)).astype(np.float)
    predicted = (np.array(predicted)).astype(np.float)
    file.close()
    return target, predicted

def l1_pred_puller(to_read):
    file  = open(to_read, 'r')
    text = file.readlines()
    target = []
    predicted = []
    
    for i in text:
        i = i.split()        
        target.append(i[0:3])
        predicted.append(i[3:6])
        
    target = (np.array(target)).astype(np.float)
    predicted = (np.array(predicted)).astype(np.float)
    file.close()
    return target, predicted

def l2_pred_puller(to_read):
    file  = open(to_read, 'r')
    text = file.readlines()
    target = []
    predicted = []
    
    for i in text:
        i = i.split()        
        target.append(i[0:5])
        predicted.append(i[5:10])
        
    target = (np.array(target)).astype(np.float)
    predicted = (np.array(predicted)).astype(np.float)
    file.close()
    return target, predicted

#This is clearly user dependent
def env_configure():
    os.environ['PATH'] = '/home/turner/TENSOAP/bin:/home/turner/TENSOAP/soapfast/scripts:/home/turner/TENSOAP/soapfast:/home/turner/TENSOAP/soapfast/scripts/uncertainty:/home/turner/TENSOAP/bin:/home/turner/TENSOAP/soapfast/scripts:/home/turner/TENSOAP/soapfast:/home/turner/TENSOAP/soapfast/scripts/uncertainty:/home/turner/TENSOAP/bin:/home/turner/TENSOAP/soapfast/scripts:/home/turner/TENSOAP/soapfast:/home/turner/TENSOAP/soapfast/scripts/uncertainty:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin'
    print('enviroment configured')
    return

def get_ps_param_par(inp, l, nc, rc):
    for i in l:
        if i == 1:
            
            subprocess.run('sagpr_parallel_get_PS -f ' + inp +' -lm '+ str(i) +
                       ' -o PS' + str(i) + ' -a -p -nc ' + str(nc) + 
                       ' -rc ' + str(rc) +' --imag'+ ' -nrun 10', shell=True)
        else:
            
            subprocess.run('sagpr_parallel_get_PS -f ' + inp +' -lm '+ str(i) +
                       ' -o PS' + str(i) + ' -a -p -nc ' + str(nc) + 
                       ' -rc ' + str(rc) + ' -nrun 10' , shell=True)
    print('Power spectrums created')
    return


#with params
def get_ps_param(inp, l, nc, rc):
    for i in l:
        if i == 1:
            
            subprocess.run('sagpr_get_PS -f ' + inp +' -lm '+ str(i) +
                       ' -o PS' + str(i) + ' -a -p -nc ' + str(nc) + 
                       ' -rc ' + str(rc) +' --imag', shell=True)
        else:
           
            subprocess.run('sagpr_get_PS -f ' + inp +' -lm '+ str(i) +
                       ' -o PS' + str(i) + ' -a -p -nc ' + str(nc) + 
                       ' -rc ' + str(rc) , shell=True)
    print('Power spectrums created')
    return

#default
def get_ps(inp, l, nc):
    for i in l:
        if i == 1:
            
            subprocess.run('sagpr_get_PS -f ' + inp +' -lm '+ str(i) +
                       ' -o PS' + str(i) + ' -a -p -nc ' + str(nc) + 
                       ' --imag', shell=True)
        else:
            
            subprocess.run('sagpr_get_PS -f ' + inp +' -lm '+ str(i) +
                       ' -o PS' + str(i) + ' -a -p -nc ' + str(nc)
                       , shell=True)
    print('Power spectrums created')
    return

def env_sparse(l, sp_size):
    for i in l:
        subprocess.run('sagpr_do_env_fps -p PS' + str(i)+ '_atomic_O.npy -n '
                       + str(sp_size) + ' -o FPS_' + str(i) + '_O', shell=True)
        subprocess.run('sagpr_do_env_fps -p PS' + str(i)+ '_atomic_Si.npy -n '
                       + str(sp_size) + ' -o FPS_' + str(i) + '_Si', shell=True)
    
    print('Enviromental FPS completed')
    return

def apply_env_sparse(l):
    for i in l:
        subprocess.run('sagpr_apply_env_fps -p PS' + str(i) + 
                       '_atomic_O.npy -sf FPS_' + str(i) + 
                       '_O_rows -o PS' + str(i) + '_atomic_sparse_O', shell=True)
        subprocess.run('sagpr_apply_env_fps -p PS' + str(i) + 
                       '_atomic_Si.npy -sf FPS_' + str(i) + 
                       '_Si_rows -o PS' + str(i) + '_atomic_sparse_Si', shell=True)        
    
    print('Eniromental sparisifcation applied')
    return


def kerneller(ker_exp):
    specs = ['O','Si']
    for i in specs:
        #L0
        subprocess.run('sagpr_get_kernel -ps PS0_atomic_' + i + 
                       '.npy PS0_atomic_sparse_' + i + 
                       '.npy -s NONE -z ' + str(ker_exp) + ' -o KERNEL_L0_NM_' + i, shell=True)
        subprocess.run('sagpr_get_kernel -ps PS0_atomic_sparse_' + i +
                       '.npy -s NONE -z ' + str(ker_exp) + ' -o KERNEL_L0_MM_' + i, shell=True)
        #L1
        subprocess.run('sagpr_get_kernel -ps PS1_atomic_' + i + 
                       '.npy PS1_atomic_sparse_' + i + 
                       '.npy -ps0 PS0_atomic_' + i + 
                       '.npy PS0_atomic_sparse_' + i + 
                       '.npy -s NONE -z ' + str(ker_exp) + ' -o KERNEL_L1_NM_' + i, shell=True)
        subprocess.run('sagpr_get_kernel -ps PS1_atomic_sparse_' + i + 
                       '.npy -ps0 PS0_atomic_sparse_' + i + 
                       '.npy -s NONE -z ' + str(ker_exp) + ' -o KERNEL_L1_MM_' + i, shell=True)
        #L2
        subprocess.run('sagpr_get_kernel -ps PS2_atomic_' + i + 
                       '.npy PS2_atomic_sparse_' + i + 
                       '.npy -ps0 PS0_atomic_' + i + 
                       '.npy PS0_atomic_sparse_' + i + 
                       '.npy -s NONE -z ' + str(ker_exp) + ' -o KERNEL_L2_NM_' + i, shell=True)
        subprocess.run('sagpr_get_kernel -ps PS2_atomic_sparse_' + i +
                       '.npy -ps0 PS0_atomic_sparse_' + i + 
                       '.npy -s NONE -z ' + str(ker_exp) + ' -o KERNEL_L2_MM_' + i, shell=True)
    print('kernels created')
    return

def model_trainer(spec, tr_size, reg):
    if isinstance(tr_size,float):
        #use fraction of total data points
        tr_no = int(round(tr_size * len(np.load('PS0_atomic_' + spec+'.npy')),0))
        
        subprocess.run('sagpr_train -r 2 -reg ' + reg + ' ' + reg + ' ' + 
                   reg + ' -f test2.xyz -sf ' +
                   'KERNEL_L0_NM_' + spec + '.npy ' + 'KERNEL_L0_MM_' + spec + '.npy ' +
                   'KERNEL_L1_NM_' + spec + '.npy ' + 'KERNEL_L1_MM_' + spec + '.npy ' +
                   'KERNEL_L2_NM_' + spec + '.npy ' + 'KERNEL_L2_MM_' + spec + '.npy ' +
                   '-c ' + spec + ' -p shielding -rdm ' + str(tr_no) + 
                   ' -pr -m \'pinv\'   ', shell=True)
    else:
        subprocess.run('sagpr_train -r 2 -reg ' + reg + ' ' + reg + ' ' + 
                   reg + ' -f test2.xyz -sf ' +
                   'KERNEL_L0_NM_' + spec + '.npy ' + 'KERNEL_L0_MM_' + spec + '.npy ' +
                   'KERNEL_L1_NM_' + spec + '.npy ' + 'KERNEL_L1_MM_' + spec + '.npy ' +
                   'KERNEL_L2_NM_' + spec + '.npy ' + 'KERNEL_L2_MM_' + spec + '.npy ' +
                   '-c ' + spec + ' -p shielding -rdm ' + str(tr_size) + 
                   ' -pr -m \'pinv\'   ', shell=True)
    print('Model training complete')
    
def inp_formatter(out_f, cat):
    #make file with full shield tensor
    master = open(out_f, 'w')
    keys = keys_grabber(cat)

    for i in keys:
        atoms = ase.io.read('Data/'+cat+'/' + str(i) + '.magres')
        positions = atoms.get_positions()
        no_at = str(len(positions))
        cell = ' '.join(map(str,np.concatenate(atoms.get_cell())))
        at_no = atoms.get_atomic_numbers()
        ms = atoms.get_array('ms')

        master.write(no_at + '\n')
        master.write('Lattice=\"' + cell + 
                     '\" Properties=species:S:1:pos:R:3:shielding:R:9 pbc=\"T T T\"\n')

        for l in range(int(no_at)):
            temp = []
            if at_no[l] == 8:
                temp.append('O')
            elif at_no[l] == 14:
                temp.append('Si')
            for t in positions[l]:
                temp.append(t)
            for t in np.concatenate(ms[l]):
                temp.append(t)
            temp.append('\n')
            temp = '   '.join(map(str,temp))
            master.write(temp)

    master.close() 
    return
    
