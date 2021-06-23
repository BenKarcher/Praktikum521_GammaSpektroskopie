#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 14:35:28 2021

@author: ben
"""
import numpy as np
from scipy.optimize import curve_fit as old_curve_fit, minimize
from numbers import Number
from scipy.stats import poisson
from collections import defaultdict
import matplotlib as mpl
#mpl.rcParams['figure.dpi'] = 150
import matplotlib.pyplot as plt
import os
def read(filename,skip=0,cols=None):
    result=[]
    with open (filename, "r") as file:
        for i in range(skip):
            file.readline()
        line1=file.readline()
        token="\t" if "\t"in line1 else (" " if " " in line1 else ",")
        arr1=line1.split(token)
        if cols==None:
            cols=range(len(arr1))
        for index in cols:
            result.append([float(arr1[index])])
        for line in file:
            line=line.replace("\n","")
            arr=line.split(token)
            if len(arr)==0 or ("" in arr):
                continue
            for count,index in enumerate(cols):
                result[count].append(float(arr[index]))
        return [np.array(c) for c in result]
    
def try_Number(string):
    try:
        return float(string)
    except ValueError:
        return string
    
def readConfig(filename):
    result={}
    with open (filename, "r") as file:
        for line in file:
            if "=" in line:
                line=line.split("=")
            else:
                line=line.split(":")
            if len(line)==2:
                result[line[0]]=try_Number(line[1].replace("\n",""))
    return result

def __poisson_wrapper__(args,f,x,y):
    y_hat=f(x,*args)
    P=0.5-np.abs(0.5-poisson.cdf(y,y_hat))
    return -np.sum(np.log(np.maximum(0.00000000000001,P)))

def poisson_fit(f,x,y,guess,preFit=True):
    if preFit:
        guess,cov=old_curve_fit(f,x,y,p0=guess)
    res=minimize(__poisson_wrapper__,guess,(f,x,y))
    y_hat=f(x,*res.x)
    chisq=np.average(((y-y_hat)/np.sqrt(np.abs(y_hat)))**2)
    return res.x, np.diag(res.hess_inv) , chisq

def curve_fit(f,x,y,yError=None,guess=None):
    if isinstance(yError,Number):
        yError=np.zeros(y.shape)+yError
    popt, pcov = old_curve_fit(f,x,y,sigma = yError,p0 = guess,absolute_sigma=True)
    # Compute chi square
    Nexp = f(x, *popt)
    r = y - Nexp
    chisq = np.average((r/yError)**2)
    return popt,np.sqrt(np.diag(pcov)),chisq

def plot(x,y,y2=None):
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    #ax1.set_xlabel("negative Spannung [mV]")
    #ax1.set_ylabel(r"$N_{mes}/N_{monitor}$", color=color)
    if type(y[0])in [np.ndarray,list]:
        for data in y:
            ax1.plot(x, data, '-')    
    else:
        ax1.plot(x, y, '-', label="daten",color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    if not y2 is None:
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        if type(y2[0])in [np.ndarray,list]:
            for data in y2:
                ax2.plot(x, data, '-')    
        else:
            ax2.plot(x, y2, '-', label="daten",color=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    #plt.title(r"Schwellenkurve von $Z_{12}$ mit Koinzidenz")
    plt.show()