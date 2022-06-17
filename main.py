"""
import os
RLfile = open('update_nonInv_Beams.m', 'r')
RLtarget = open('update_nonInv_Beams.m.bak', 'w+')
lines=RLfile.readlines()
RLfile.close()
for line in lines:
    tmp=line.split()
    if len(tmp)==0 or tmp[0]!='kStiff':
        RLtarget.write(line)
    else:
        k=float(tmp[2])
        RLtarget.write('kStiff = '+str(k/100)+' ;\n')
RLtarget.close()
os.remove(RLfile.name)
newname=RLtarget.name.replace('.bak', '')
os.rename(RLtarget.name, newname)


RLfile = open('help_Me_Restart.m', 'r')
RLtarget = open('help_Me_Restart.m.bak', 'w+')
lines = RLfile.readlines()
RLfile.close()
for line in lines:
    tmp = line.split()
    if len(tmp) == 0 or tmp[0] != 'ctsave':
        RLtarget.write(line)
    else:
        RLtarget.write('ctsave = 8 ;\n')
RLtarget.close()
os.remove(RLfile.name)
newname = RLtarget.name.replace('.bak', '')
os.rename(RLtarget.name, newname)


RLfile = open('input2d', 'r')
RLtarget = open('input2d.bak', 'w+')
lines = RLfile.readlines()
RLfile.close()
for line in lines:
    tmp = line.split()
    if len(tmp) == 0 or tmp[0] != 'Restart_Flag':
        RLtarget.write(line)
    else:
        RLtarget.write('Restart_Flag = 1\n')
RLtarget.close()
os.remove(RLfile.name)
newname = RLtarget.name.replace('.bak', '')
os.rename(RLtarget.name, newname)

os.rename('viz_IB2d', 'viz_IB2d_1')
os.rename('hier_IB2d_data', 'hier_IB2d_data_1')



env = Flow_Field()
done = False
while not done:
    action = env.action_space[4]
    observation_, reward, done = env.step(action)

"""
from field_env import Flow_Field
import random
import numpy as np



env = Flow_Field()
done = False
while not done:
    action = env.action_space[4]
    observation_, reward, done = env.step(action)
#print(env.dt)


#file=open('input2d','r')
#file.close()

