'''
1stage opamp dataset generation
Used for predicting OPAMP MOSFET sizes depending on performances
2022 Junsang Yoo
'''
# todo: some consistancy to variable name

## imports
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# pyspice import
import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
from PySpice.Spice.Library import SpiceLibrary

# os.chdir(os.path.dirname(__file__))     # change directory to where current file is
spice_library = SpiceLibrary('./')      # use TSMC 180nm library file (measured)

# import simulation function
from simulation import simulate

## OPAMP design parameters
'''
  --------------------------------
  |                |             |
  |                s             s
  |               M3 g---------g M4
  |                d      |      d
{ibais}            |      |      |
  |                |-----(3)     |-----OUT
 \ /               |             |
  |                d             d
  |              g M1           M2 g
  |                s             s
  |                |             |
  |                ------(2)------
  |------------           |
  |           |           |
  d           |           d
 M6 g--------(1)--------g M5
  s                       s
  |                       |
  -------------------------

This uses TSMC 180nm device
For 180nm devices, the working voltage should be 1.8V
minimum Length: 180nm
minimum width: 220nm
constraints for differential pair: W1=W2, W3=W4
'''

VDD=1.8; VSS=0
ibias=20e-6
Lmin=180e-9; Wmin=220e-9
W_samp=np.array([10e-6,10e-6,0.42e-6,0.42e-6,10e-6,5e-6])   # sample W1~W6
L_samp=np.array([Lmin,Lmin,Lmin,Lmin,1e-6,1e-6])            # sample L1~L6
MOScnt=6  # total number of MOSFET

## Circuit definition
opamp = Circuit('opamp')
opamp.include(spice_library['nch'])   # this includes library file that has nch in it

# OPAMP
opamp.I(1,'DD',3,ibias)     # current source
M1=opamp.M(1,1,'INP',2,2,model='nch',l=L_samp[0],w=W_samp[0])          # differential pair
M2=opamp.M(2,'OUT','INN',2,2,model='nch',l=L_samp[1],w=W_samp[1])      # differential pair
M3=opamp.M(3,1,1,'DD','DD',model='pch',l=L_samp[2],w=W_samp[2])        # active load
M4=opamp.M(4,'OUT',1,'DD','DD',model='pch',l=L_samp[3],w=W_samp[3])    # active load
M5=opamp.M(5,2,3,'SS','SS',model='nch',l=L_samp[4],w=W_samp[4])        # current source
M6=opamp.M(6,3,3,'SS','SS',model='nch',l=L_samp[5],w=W_samp[5])        # current mirror
opamp.C(1,'OUT',0,100e-15)  # loading cap

print(opamp)

# test simulate function
perf=simulate(opamp)
print(perf)

## CSV Generation Parameters
W1range=np.round(np.arange(1e-6,11e-6,1e-6),10)
W3range=np.round(np.arange(0.3e-6,5e-6,0.3e-6),10)
W5range=np.round(np.arange(2e-6,11e-6,2e-6),10)
W6range=W5range

## Dataset Generation
itercnt=0
with open('dataset.csv', 'w') as f:
    # title
    f.write('w1,w2,w3,w4,w5,w6,')
    for key in perf.keys():
        f.write(key+',')
    f.write('\n')
    for w1 in W1range:
        for w3 in W3range:
            for w5 in W5range:
                for w6 in W6range:
                    itercnt+=1
                    w2=w1; w4=w3
                    M1.width=w1
                    M2.width=w2
                    M3.width=w3
                    M4.width=w4
                    M5.width=w5
                    M6.width=w6

                    if (1+itercnt)%100==0:
                        print(f'sim {itercnt+1}')
                    try:
                        perf=simulate(opamp)
                        f.write(f'{w1},{w2},{w3},{w4},{w5},{w6},')
                        for val in perf.values():
                            if type(val)==list:
                                f.write(str(np.sum(val))+',')
                            else:
                                f.write(f'{val:.5e},')
                        f.write('\n')
                    except:
                        print(f'#{itercnt} sim failed')

