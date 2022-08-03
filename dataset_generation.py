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

# pyspice import
import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
from PySpice.Spice.Library import SpiceLibrary

# os.chdir(os.path.dirname(__file__))
spice_library = SpiceLibrary('./')      # use TSMC 180nm library file (measured)

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
'''

VDD=1.8; VSS=0
ibias=20e-6
Lmin=180e-9; Wmin=220e-9
W=np.array([10e-6,10e-6,0.42e-6,0.42e-6,10e-6,5e-6])    # W1~W6
L=np.array([Lmin,Lmin,Lmin,Lmin,1e-6,1e-6])     # L1~L6
MOScnt=6  # total number of MOSFET

## Circuit definition
opamp = Circuit('opamp')
opamp.include(spice_library['nch'])   # this includes library file that has nch in it

# OPAMP
opamp.I(1,'DD',3,ibias)     # current source
opamp.M(1,1,'INP',2,2,model='nch',l=L[0],w=W[0])          # differential pair
opamp.M(2,'OUT','INN',2,2,model='nch',l=L[1],w=W[1])      # differential pair
opamp.M(3,1,1,'DD','DD',model='pch',l=L[2],w=W[2])        # active load
opamp.M(4,'OUT',1,'DD','DD',model='pch',l=L[3],w=W[3])    # active load
opamp.M(5,2,3,'SS','SS',model='nch',l=L[4],w=W[4])        # current source
opamp.M(6,3,3,'SS','SS',model='nch',l=L[5],w=W[5])        # current mirror
opamp.C(1,'OUT',0,100e-15)  # loading cap

print(opamp)

## Simulation Function
def simulate(opamp, W, L):
    '''
    opamp simulation function
    arguments:
        opamp: opamp circuit to simulate. will be cloned and than used.
        W: array of widths of MOSFETs in opamp
        L: array of lengths of MOSFETs in opamp
    returns:
        dictionary of performances
    '''
    perf={}
    circuit=opamp.clone()

    # ************************
    # Operating Point Analysis
    # ************************
    tb_op=opamp.clone(title='testbench operating point')

    # Voltage Sources
    tb_op.V(1,'DD',0,VDD)       # VDD
    tb_op.V(2,'SS',0,0)         # VSS

    # Input Voltage
    tb_op.R(1,'DD','INP',50@u_kOhm)
    tb_op.R(2,'INN','SS',50@u_kOhm)
    tb_op.R(3,'INP','INN',0@u_Ohm)     # added to regard nodes INP and INN as same node

    # Params to Save
    simulator= tb_op.simulator(temperature = 25, nominal_temperature =25)
    internal_params='all'
    for i in range(MOScnt):
        internal_params+=(f' @m{int(i+1)}[vds] @m{int(i+1)}[vdsat] @m{int(i+1)}[vth] @m{int(i+1)}[vgs] @m{int(i+1)}[id] @m{int(i+1)}[gm] @m{int(i+1)}[gds]')
    simulator.save_internal_parameters(internal_params)

    # Operating Point simulation
    tb_op_sim=simulator.operating_point()
    perf['op_vout']=float(tb_op_sim.out)

    # saturation check
    satlist=[]
    for i in range(MOScnt):
        cond1=float(tb_op_sim[f'@m{int(i+1)}[vds]'])>float(tb_op_sim[f'@m{int(i+1)}[vdsat]'])
        cond2=float(tb_op_sim[f'@m{int(i+1)}[vgs]'])>float(tb_op_sim[f'@m{int(i+1)}[vth]'])
        if cond1 and cond2:
            satlist.append(1)
        else:
            satlist.append(0)
    perf['op_satlist']=satlist

    # gain calculation
    ro2=1/float(tb_op_sim['@m2[gds]'])
    ro4=1/float(tb_op_sim['@m4[gds]'])
    Ro=ro2*ro4/(ro2+ro4)
    op_gain=float(tb_op_sim['@m2[gm]'])*Ro
    op_gaindB=20*np.log10(op_gain)
    perf['op_gain']=op_gain
    perf['op_gain_db']=op_gaindB

    # ICMR
    op_minVINN=VSS+float(tb_op_sim['@m5[vdsat]'])+float(tb_op_sim['@m2[vgs]'])
    op_maxVINN=VDD-float(tb_op_sim['@m4[vdsat]'])-float(tb_op_sim['@m2[vdsat]'])+float(tb_op_sim['@m2[vgs]'])
    perf['op_inmin']=op_minVINN
    perf['op_inmax']=op_maxVINN
    # todo: perf['op_icmr']

    # Output Range
    op_minOUT=VSS+float(tb_op_sim['@m5[vdsat]'])+float(tb_op_sim['@m2[vds]'])
    op_maxOUT=VDD-float(tb_op_sim['@m4[vdsat]'])
    perf['op_outmin']=op_minOUT
    perf['op_outmax']=op_maxOUT
    # todo: perf['op_outrange']

    # ***********
    # DC Analysis
    # ***********
    tb_dc=opamp.clone(title='testbench dc')

    # Voltage Sources
    tb_dc.V(1,'DD',0,VDD)       # VDD
    tb_dc.V(2,'SS',0,0)         # VSS
    tb_dc.VCVS(1,'INP','COMM','DIFF',0,0.5)
    tb_dc.VCVS(2,'INN','COMM','DIFF',0,-0.5)
    tb_dc.V(3,'DIFF',0,0)
    tb_dc.V(4,'COMM',0,0.9)

    # Differential Input
    simulator=tb_dc.simulator(temperature=25, nominal_temperature=25)
    tb_dc_sim=simulator.dc(V3=slice(-VDD,VDD,0.01))
    outdiff_wave=np.diff(tb_dc_sim['out'])/0.01
    outdiff=[]
    for i in outdiff_wave:
        outdiff.append(i.value)
    outdiff=np.array(outdiff)
    outmaxidx=np.where(outdiff>1)[0][-1];   dc_outmax=tb_dc_sim['out'][outmaxidx].value
    outminidx=np.where(outdiff>1)[0][0];    dc_outmin=tb_dc_sim['out'][outminidx].value
    perf['dc_outmin']=dc_outmin
    perf['dc_outmax']=dc_outmax
    # todo: perf['dc_outrange']

    # Common Input
    tb_dc_sim=simulator.dc(V4=slice(0,VDD,0.01))
    incomm_wave=np.diff(tb_dc_sim['INP']-tb_dc_sim['2'])/0.01   # Vgs of M1
    incomm=[]
    for i in incomm_wave:
        incomm.append(i.value)
    incomm=np.array(incomm)
    inmaxidx=np.where(incomm<0.5)[0][-1];   dc_inmax=tb_dc_sim['INP'][inmaxidx].value
    inminidx=np.where(incomm<0.5)[0][0];    dc_inmin=tb_dc_sim['INP'][inminidx].value
    perf['dc_inmin']=dc_inmin
    perf['dc_outmax']=dc_outmax

    # ******************
    # Transient Analysis
    # ******************
    tb_tran=opamp.clone(title='testbench tran')

    # Voltage Sources
    tb_tran.V(1,'DD',0,VDD)       # VDD
    tb_tran.V(2,'SS',0,0)         # VSS
    tb_tran.VCVS(1,'INP','COMM','DIFF',0,0.5)
    tb_tran.VCVS(2,'INN','COMM','DIFF',0,-0.5)
    tran_vsource=tb_tran.SinusoidalVoltageSource(3,'DIFF',0,amplitude=20@u_mV,frequency=100@u_Hz)
    tb_tran.V(4,'COMM',0,0.9)

    # Small Signal
    simulator=tb_tran.simulator(temperature=25,nominal_temperature=25)
    tb_tran_sim=simulator.transient(step_time=tran_vsource.period/200,end_time=tran_vsource.period*2)
    voutmean=np.mean(tb_tran_sim['out'])
    voutmax=np.max(tb_tran_sim['out'])
    tran_gain=float(voutmax-voutmean)/20e-3
    tran_gaindB=20*np.log10(tran_gain)
    perf['tran_gain']=tran_gain
    perf['tran_gain_db']=tran_gaindB

    # ***********
    # AC Analysis
    # ***********
    tb_ac=opamp.clone(title='testbench ac')

    # Voltage Sources
    tb_ac.V(1,'DD',0,VDD)       # VDD
    tb_ac.V(2,'SS',0,0)         # VSS
    tb_ac.VCVS(1,'INP','COMM','DIFF',0,0.5)
    tb_ac.VCVS(2,'INN','COMM','DIFF',0,-0.5)
    tb_ac.SinusoidalVoltageSource(3,'DIFF',0,amplitude=1@u_mV,frequency=1@u_kHz)
    tb_ac.V(4,'COMM',0,0.9)

    # gain
    simulator=tb_ac.simulator(temperature=25,nominal_temperature=25)
    tb_ac_sim=simulator.ac(start_frequency=1@u_Hz, stop_frequency=10@u_GHz, number_of_points=21,  variation='dec')
    ac_vout=tb_ac_sim['out']
    ac_gain=float(np.abs(ac_vout)[0])
    ac_gaindB=20*np.log10(ac_gain)
    perf['ac_gain']=ac_gain
    perf['ac_gain_db']=ac_gaindB

    # GBP and Phase Margin
    for idx0db in range(len(tb_ac_sim['out'])):
        if np.abs(tb_ac_sim['out'])[idx0db]<=1:
            break
    ac_GBP=float(tb_ac_sim.frequency[idx0db])
    ac_PM=np.angle(tb_ac_sim['out'],deg=True)[idx0db]+180
    perf['ac_GBP']=ac_GBP

    return perf

## CSV generation
