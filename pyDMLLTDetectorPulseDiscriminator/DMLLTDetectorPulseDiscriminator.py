# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 13:25:01 2019

@author: Danny Petschke
@email:  danny.petschke@uni-wuerzburg.de

"""

#*************************************************************************************************
#**
#** DMLLTDetectorPulseDiscriminator v1.0 (29.03.2019)
#**
#**
#** Copyright (c) 2019-2021 Danny Petschke. All rights reserved.
#** 
#** Redistribution and use in source and binary forms, with or without modification, 
#** are permitted provided that the following conditions are met:
#**
#** 1. Redistributions of source code must retain the above copyright notice, 
#**    this list of conditions and the following disclaimer.
#**
#** 2. Redistributions in binary form must reproduce the above copyright notice, 
#**    this list of conditions and the following disclaimer in the documentation 
#**    and/or other materials provided with the distribution.
#**
#** 3. Neither the name of the copyright holder "Danny Petschke" nor the names of its  
#**    contributors may be used to endorse or promote products derived from this software  
#**    without specific prior written permission.
#**
#**
#** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS 
#** OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF 
#** MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
#** COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
#** EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
#** SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) 
#** HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR 
#** TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
#** EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#**
#** contact:      danny.petschke@uni-wuerzburg.de
#**
#** researchGate: https://www.researchgate.net/profile/Danny_Petschke
#** linkedIn:     https://de.linkedin.com/in/petschkedanny
#**
#*************************************************************************************************

import sys
import os
import struct
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

from joblib import dump, load
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV

from scipy.signal import medfilt
from scipy.interpolate import CubicSpline

"""

 This class holds all information, which are required to TRAIN and TEST a machine('s classifier).
 Moreover, it provides (re)storing the learned machine's classifier from a file (*.joblib).
 
"""

class DMachineParams():
    # baseline correction
    m_correctForBaseline = True  
    m_startCell          = 10 
    m_cellRegion         = 150
    
    # median filter
    m_medianFilter       = True 
    m_windowSize         = 5
    
    m_classifier          = CalibratedClassifierCV(GaussianNB(), cv=2, method='isotonic')
    
    def __init__(self, 
                 correctForBaseline = True, 
                 startCell          = 10, 
                 cellRegion         = 150, 
                 medianFilter       = True, 
                 windowSize         = 5, 
                 classifier         = CalibratedClassifierCV(GaussianNB(), cv=2, method='isotonic')):
        self.m_classifier         = deepcopy(classifier)
        
        self.m_correctForBaseline = correctForBaseline
        self.m_startCell          = startCell
        self.m_cellRegion         = cellRegion
        self.m_medianFilter       = medianFilter
        self.m_windowSize         = windowSize
        
    def load(self, fileNameAndPath = '/name'):
        dumpList = load(fileNameAndPath + '.joblib')
        
        self.m_classifier         = deepcopy(dumpList[0])
        
        self.m_correctForBaseline = dumpList[1]
        self.m_startCell          = dumpList[2]
        self.m_cellRegion         = dumpList[3]
        self.m_medianFilter       = dumpList[4]
        self.m_windowSize         = dumpList[5]
        
    def save(self, fileNameAndPath = '/name'):
        dumpList = [self.m_classifier, self.m_correctForBaseline, self.m_startCell, self.m_cellRegion, self.m_medianFilter, self.m_windowSize]
        
        dump(dumpList, fileNameAndPath + '.joblib')
        
    def copy(self):
        machineParams                      = DMachineParams()
        
        machineParams.m_classifier         = deepcopy(self.m_classifier)
        
        machineParams.m_correctForBaseline = self.m_correctForBaseline
        machineParams.m_startCell          = self.m_startCell
        machineParams.m_cellRegion         = self.m_cellRegion
        machineParams.m_medianFilter       = self.m_medianFilter
        machineParams.m_windowSize         = self.m_windowSize
        
        return machineParams
        
    def debug(self):
        print("---------------------------------------------------------------")
        print("ML classifier:         {0}".format(self.m_classifier))
        print("")
        print("baseline correction?:  {0}".format(self.m_correctForBaseline))
        print("region:                [{0}:{1}]".format(self.m_startCell, self.m_cellRegion))
        print("")
        print("median filter?:        {0}".format(self.m_medianFilter))
        print("window size:           {0}".format(self.m_windowSize))
        print("---------------------------------------------------------------")
        
"""

 This function reads the required information defined in the header (c-type struct) 
 of each pulse-stream file:
     
 For more information regarding pulse-streaming see the C++ API: DPulseStreamAPI.h/.cpp
 
 The struct is designed as following:
    
    struct {
            version         = type(uint32)
            __              = type(uint32)
            sweep [ns]      = type(double) --> readout range
            freq. [GHz]     = type(double) --> sampling frequency
            sampling points = type(int32)  --> number of sampling points per readout
            __              = type(uint32)
    }
 
 Note: ONLY the number of 'sampling points' is required for the framework functionality. 
   
"""

def readHeader(file):
    __ = struct.unpack('i', file.read(4))[0] #unused
    __ = struct.unpack('i', file.read(4))[0] #unused
    
    sweepInNanoseconds = struct.unpack('d', file.read(8))[0] #unused
    frequencyInGHz     = struct.unpack('d', file.read(8))[0] #unused
    numberOfCells      = struct.unpack('i', file.read(4))[0]

    __                 = struct.unpack('i', file.read(4))[0] #necessary to fill 32bytes (struct)
   
    return numberOfCells, sweepInNanoseconds, frequencyInGHz

"""

 This function reads the time [ns] and voltage [mV] traces of a single detector pulse in 
 a given pulse-stream file 'file'.
     
"""

def readPulse(file, numberOfCells):
    time = np.zeros(numberOfCells)
    volt = np.zeros(numberOfCells)
        
    aborted   = False
    resort    = False
    priorTime = -1e10
    
    for i in range (0, numberOfCells):
        byteChunk = file.read(4)
        
        if not byteChunk:
            aborted = True
            break
        
        if (time[i] < priorTime):
            resort = True
            
        time[i] = struct.unpack('f', byteChunk)[0]
        priorTime = time[i]
        
    if aborted:
        time = np.zeros(0)
        
    if resort:
        time = np.sort(time)
        
    aborted = False
             
    for i in range (0, numberOfCells):
        byteChunk = file.read(4)
        
        if not byteChunk:
            aborted = True
            break
        
        volt[i] = struct.unpack('f', byteChunk)[0]
        
    if aborted:
        volt = np.zeros(0)
        
    return time, volt

"""

 This function normalizes the pulse shape for TRAINing/TESTing and PREDICTING:
     
 (1) by its integrated area over the entire time frame (number of cells) with respect to the zero baseline and then
 (2) by its absolute value of maximum/minimum amplitude that: max(amplitude) = 1.0. 
     
"""

def normalizeData(voltage, numberOfCells, polarity):
    arg = 0
    
    valid = True
    
    if not polarity: #negative
        arg = np.argmin(voltage)
    else:            #positive
        arg = np.argmax(voltage)
        
    minMaxValue = voltage[arg]
    minMaxArg   = arg    
    
    # safety region (may adjust this region for your purposes)
    if minMaxArg < 0.02*numberOfCells or minMaxArg > 0.92*numberOfCells:
        valid = False
        
        np.roll(np.roll(voltage, numberOfCells-arg), (int)(numberOfCells/2)), minMaxValue, minMaxArg, valid
                   
    voltage /= np.abs(sum(voltage))   
    
    if not polarity: #negative
        arg = np.argmin(voltage)
    else:            #positive
        arg = np.argmax(voltage)
        
    if arg < 0 or arg >= numberOfCells:
        valid = False
        
        return voltage, minMaxValue, minMaxArg, valid
    
    if voltage[arg] == 0.0: # prevent by zero division
        valid = False
        
        return voltage, 0.0, minMaxArg, valid
    
    voltage /= voltage[arg]
    
    return np.roll(np.roll(voltage, numberOfCells-arg), (int)(numberOfCells/2)), minMaxValue, minMaxArg, valid

"""

 This function calculates the time difference, i.e. the lifetime between two detector pulses using the constant fraction (CF) principle.
 
"""

def calcLifetime(xStart                  = [], 
                 yStart                  = [], 
                 xStop                   = [], 
                 yStop                   = [],
                 cfd_level_start         = 25.0,
                 cfd_level_stop          = 25.0,
                 amplitudeStart          = 0.0,
                 amplitudeStop           = 0.0,
                 isPositivePolarity      = False,
                 cubicSpline             = True,
                 cubicSplineRenderPoints = 200):
    timeStart = 0.0
    timeStop  = 0.0
    
    cfdVoltageStart = cfd_level_start*0.01*amplitudeStart
    cfdVoltageStop  = cfd_level_stop *0.01*amplitudeStop
        
    rejectStart = True
    rejectStop  = True
    
    # using cubic spline interpolation?
    if cubicSpline:
        #negative polarity
        if not isPositivePolarity: 
            argStart = np.argmin(yStart)  
            argStop  = np.argmin(yStop)  
            
            # start branch
            index = -1
            
            for i in range(argStart, 0, -1):
                if yStart[i] <= cfdVoltageStart and yStart[i-1] >= cfdVoltageStart:
                    index = i
                    break
            
            if not index <= 2:
                xRenderPoints = np.zeros(cubicSplineRenderPoints)
                yCubicValues  = np.zeros(cubicSplineRenderPoints)
                
                time_lower = xStart[index-1]
                time_upper = xStart[index]
                
                timeIncr = np.abs(time_upper - time_lower)/float(cubicSplineRenderPoints)
                
                spline = CubicSpline(xStart, yStart)
                
                for cubic in range(0, cubicSplineRenderPoints):
                    xRenderPoints[cubic] = time_lower + float(cubic)*timeIncr
                    yCubicValues[cubic]  = spline(xRenderPoints[cubic])
                    
                for i in range(cubicSplineRenderPoints-1, 0, -1):
                    if yCubicValues[i] <= cfdVoltageStart and yCubicValues[i-1] >= cfdVoltageStart:
                        slope     = (yCubicValues[i-1]-yCubicValues[i])/(xRenderPoints[i-1]-xRenderPoints[i])
                        intercept = yCubicValues[i] - slope*xRenderPoints[i]
                        
                        if slope >= 0 or i <= 2:
                            break
                            
                        timeStart   = (cfdVoltageStart - intercept)/slope
                        rejectStart = False
            
            # stop branch
            index = -1
            
            for i in range(argStop, 0, -1):
                if yStop[i] <= cfdVoltageStop and yStop[i-1] >= cfdVoltageStop:
                    index = i
                    break
            
            if not index <= 2:
                xRenderPoints = np.zeros(cubicSplineRenderPoints)
                yCubicValues  = np.zeros(cubicSplineRenderPoints)
                
                time_lower = xStop[index-1]
                time_upper = xStop[index]
                
                timeIncr = np.abs(time_upper - time_lower)/float(cubicSplineRenderPoints)
                   
                spline = CubicSpline(xStop, yStop)
                
                for cubic in range(0, cubicSplineRenderPoints):
                    xRenderPoints[cubic] = time_lower + float(cubic)*timeIncr
                    yCubicValues[cubic]  = spline(xRenderPoints[cubic])
                    
                for i in range(cubicSplineRenderPoints-1, 0, -1):
                    if yCubicValues[i] <= cfdVoltageStop and yCubicValues[i-1] >= cfdVoltageStop:
                        slope     = (yCubicValues[i-1]-yCubicValues[i])/(xRenderPoints[i-1]-xRenderPoints[i])
                        intercept = yCubicValues[i] - slope*xRenderPoints[i]
                        
                        if slope >= 0 or i <= 2:
                            break
                            
                        timeStop   = (cfdVoltageStop - intercept)/slope
                        rejectStop = False
           
        #positive polarity         
        else:
            argStart = np.argmax(yStart)  
            argStop  = np.argmax(yStop)  
            
            # start branch
            index = -1
            
            for i in range(argStart, 0, -1):
                if yStart[i] >= cfdVoltageStart and yStart[i-1] <= cfdVoltageStart:
                    index = i
                    break
            
            if not index <= 2:
                xRenderPoints = np.zeros(cubicSplineRenderPoints)
                yCubicValues  = np.zeros(cubicSplineRenderPoints)
                
                time_lower = xStart[index-1]
                time_upper = xStart[index]
                
                timeIncr = np.abs(time_upper - time_lower)/float(cubicSplineRenderPoints)
                
                spline = CubicSpline(xStart, yStart)
                
                for cubic in range(0, cubicSplineRenderPoints):
                    xRenderPoints[cubic] = time_lower + float(cubic)*timeIncr
                    yCubicValues[cubic]  = spline(xRenderPoints[cubic])
                
                for i in range(cubicSplineRenderPoints-1, 0, -1):
                    if yCubicValues[i] >= cfdVoltageStart and yCubicValues[i-1] <= cfdVoltageStart:
                        slope     = (yCubicValues[i-1]-yCubicValues[i])/(xRenderPoints[i-1]-xRenderPoints[i])
                        intercept = yCubicValues[i] - slope*xRenderPoints[i]
                        
                        if slope <= 0 or i <= 2:
                            break
                            
                        timeStart   = (cfdVoltageStart - intercept)/slope
                        rejectStart = False
            
            # stop branch
            index = -1
            
            for i in range(argStop, 0, -1):
                if yStop[i] >= cfdVoltageStop and yStop[i-1] <= cfdVoltageStop:
                    index = i
                    break
            
            if not index <= 2:
                xRenderPoints = np.zeros(cubicSplineRenderPoints)
                yCubicValues  = np.zeros(cubicSplineRenderPoints)
                
                time_lower = xStop[index-1]
                time_upper = xStop[index]
                
                timeIncr = np.abs(time_upper - time_lower)/float(cubicSplineRenderPoints)
                
                spline = CubicSpline(xStop, yStop)
                
                for cubic in range(0, cubicSplineRenderPoints):
                    xRenderPoints[cubic] = time_lower + float(cubic)*timeIncr
                    yCubicValues[cubic]  = spline(xRenderPoints[cubic])
                
                for i in range(cubicSplineRenderPoints-1, 0, -1):
                    if yCubicValues[i] >= cfdVoltageStop and yCubicValues[i-1] <= cfdVoltageStop:
                        slope     = (yCubicValues[i-1]-yCubicValues[i])/(xRenderPoints[i-1]-xRenderPoints[i])
                        intercept = yCubicValues[i] - slope*xRenderPoints[i]
                        
                        if slope <= 0 or i <= 2:
                            break
                            
                        timeStop   = (cfdVoltageStop - intercept)/slope
                        rejectStop = False
    
    # linear interpolation                 
    else:
        #negative polarity
        if not isPositivePolarity: 
            argStart = np.argmin(yStart)  
            argStop  = np.argmin(yStop)  
            
            for i in range(argStart, 0, -1):
                if yStart[i] <= cfdVoltageStart and yStart[i-1] >= cfdVoltageStart:
                    slope     = (yStart[i-1]-yStart[i])/(xStart[i-1]-xStart[i])
                    intercept = yStart[i] - slope*xStart[i]
                    
                    if slope >= 0 or i <= 2:
                        break
                        
                    timeStart   = (cfdVoltageStart - intercept)/slope
                    rejectStart = False       
                                            
            for i in range(argStop, 0, -1):
                if yStop[i] <= cfdVoltageStop and yStop[i-1] >= cfdVoltageStop:
                    slope     = (yStop[i-1]-yStop[i])/(xStop[i-1]-xStop[i])
                    intercept = yStop[i] - slope*xStop[i]
                    
                    if slope >= 0 or i <= 2:
                        break
                    
                    timeStop   = (cfdVoltageStop - intercept)/slope
                    rejectStop = False
        #positive polarity
        else:            
            argStart = np.argmax(yStart)  
            argStop  = np.argmax(yStop)  
            
            for i in range(argStart, 0, -1):
                if yStart[i] >= cfdVoltageStart and yStart[i-1] <= cfdVoltageStart:
                    slope     = (yStart[i-1]-yStart[i])/(xStart[i-1]-xStart[i])
                    intercept = yStart[i] - slope*xStart[i]
                    
                    if slope <= 0 or i <= 2:
                        break
                    
                    timeStart   = (cfdVoltageStart - intercept)/slope
                    rejectStart = False
                                
            for i in range(argStop, 0, -1):
                if yStop[i] >= cfdVoltageStop and yStop[i-1] <= cfdVoltageStop:
                    slope     = (yStop[i-1]-yStop[i])/(xStop[i-1]-xStop[i])
                    intercept = yStop[i] - slope*xStop[i]
                    
                    if slope <= 0 or i <= 2:
                        break
                    
                    timeStop   = (cfdVoltageStop - intercept)/slope
                    rejectStop = False
                           
    return 1000.0*(timeStop - timeStart), (rejectStart or rejectStop)

"""

 This function can be used to TRAIN and TEST a machine's classifier from only ONE data set of streamed 
 pulses 'fileNameCorrectPulses' and 'fileNameRejectPulses'.

 If 'splitAfterNPulses' == -1 (default), the pulses in the respective 
 pulse streams 'fileNameCorrectPulses' and 'fileNameRejectPulses' are splitted after 50% 
 of pulses read and used for the TRAINing process. The remaining 50% of streamed pulses are 
 considered for TESTing the TRAINed machine's classifier.
 
 return: 
     
     (1) DMachineParams() from the learned machine and 
     (2) the prediction accuracy [0.0-1.0].
 
"""

def splitTrainAndTest(fileNameCorrectPulses = '/correct', 
                      fileNameRejectPulses  = '/reject',  
                      isPositivePolarity    = False,
                      splitAfterNPulses     = -1,
                      machineInput          = DMachineParams()):
    mlInput = machineInput.copy()
    
    fileSizeTrue  = os.path.getsize(fileNameCorrectPulses)
    fileSizeFalse = os.path.getsize(fileNameRejectPulses)
    
    # train data
    x_array_train   = []
    y_array_train   = []
    
    # test data
    x_array_test    = []
    y_array_test    = []
    
    readBytes       = 32 #header offset
    
    numberOfPulses_train = 0
    numberOfPulses_test  = 0
    
    # (1) REJECT pulses:
    with open(fileNameRejectPulses, "rb") as streamFile:
        numberOfCells, sweepInNanoseconds, frequencyInGHz = readHeader(streamFile)
        
        pulseBytes = 2*numberOfCells*4
        numberOfPulses = (fileSizeFalse - readBytes)/pulseBytes
        
        if not (numberOfPulses%2): # even
            numberOfPulses_train = numberOfPulses/2
        else: # odd
            numberOfPulses_train = (numberOfPulses-1)/2
                                   
        numberOfPulses_test = numberOfPulses_train
                         
        if not splitAfterNPulses == -1:
            numberOfPulses_train = splitAfterNPulses
            numberOfPulses_test  = numberOfPulses_train
        
        pulseCounter = 0
    
        while True:
            __, pulse = readPulse(streamFile, numberOfCells)
                
            if not len(pulse):
                break
            
            # apply median filter?:
            if mlInput.m_medianFilter:
                pulse = medfilt(pulse, mlInput.m_windowSize)
                    
            # correct for baseline?:
            if mlInput.m_correctForBaseline:
                stddev_pre  = np.std(pulse[mlInput.m_startCell:mlInput.m_cellRegion])
                stddev_post = np.std(pulse[numberOfCells-1-mlInput.m_startCell-mlInput.m_cellRegion:numberOfCells-1-mlInput.m_startCell])
                    
                mean = 0.0
                    
                if np.abs(stddev_pre) < np.abs(stddev_post):
                    mean = np.mean(pulse[mlInput.m_startCell:mlInput.m_cellRegion])
                else:
                    mean = np.mean(pulse[numberOfCells-1-mlInput.m_startCell-mlInput.m_cellRegion:numberOfCells-1-mlInput.m_startCell])
                           
                pulse -= mean
                    
            readBytes += pulseBytes
            
            # normalize pulse data
            voltage_norm, __, __, valid = normalizeData(pulse, numberOfCells, isPositivePolarity)
                
            if not valid:
                continue
            
            pulseCounter += 1
                
            if pulseCounter <= numberOfPulses_train:
                x_array_train.append(voltage_norm)
                y_array_train.append(0) # << FALSE (0) means 'bad' pulses (REJECT)
            elif pulseCounter > numberOfPulses_train and pulseCounter <= numberOfPulses_train + numberOfPulses_test:
                x_array_test.append(voltage_norm)
                y_array_test.append(0) # << FALSE (0) means 'bad' pulses (REJECT)
            else:
                break
                
        streamFile.close()
        
    readBytes = 32 #header offset
    
    # (2) CORRECT pulses:
    with open(fileNameCorrectPulses, "rb") as streamFile:
        numberOfCells, sweepInNanoseconds, frequencyInGHz = readHeader(streamFile)
        
        pulseBytes = 2*numberOfCells*4
        numberOfPulses = (fileSizeTrue - readBytes)/pulseBytes
        
        if not (numberOfPulses%2): # even
            numberOfPulses_train = numberOfPulses/2
        else: # odd
            numberOfPulses_train = (numberOfPulses-1)/2
                                   
        numberOfPulses_test = numberOfPulses_train
                         
        if not splitAfterNPulses == -1:
            numberOfPulses_train = splitAfterNPulses
            numberOfPulses_test  = numberOfPulses_train
        
        pulseCounter = 0
          
        while True:
            __, pulse = readPulse(streamFile, numberOfCells)
                
            if not len(pulse):
                break
                
            # apply median filter?:
            if mlInput.m_medianFilter:
                pulse = medfilt(pulse, mlInput.m_windowSize)
                    
            # correct for baseline?:
            if mlInput.m_correctForBaseline:
                stddev_pre  = np.std(pulse[mlInput.m_startCell:mlInput.m_cellRegion])
                stddev_post = np.std(pulse[numberOfCells-1-mlInput.m_startCell-mlInput.m_cellRegion:numberOfCells-1-mlInput.m_startCell])
                    
                mean = 0.0
                    
                if np.abs(stddev_pre) < np.abs(stddev_post):
                    mean = np.mean(pulse[mlInput.m_startCell:mlInput.m_cellRegion])
                else:
                    mean = np.mean(pulse[numberOfCells-1-mlInput.m_startCell-mlInput.m_cellRegion:numberOfCells-1-mlInput.m_startCell])
                           
                pulse -= mean
                    
            readBytes += pulseBytes
                
            # normalize pulse data
            voltage_norm, __, __, valid = normalizeData(pulse, numberOfCells, isPositivePolarity)
                
            if not valid:
                continue
                
            pulseCounter += 1
                
            if pulseCounter <= numberOfPulses_train:
                x_array_train.append(voltage_norm)
                y_array_train.append(1) # << TRUE (1) means 'good' pulses (CORRECT)
            elif pulseCounter > numberOfPulses_train and pulseCounter <= numberOfPulses_train + numberOfPulses_test:
                x_array_test.append(voltage_norm)
                y_array_test.append(1) # << TRUE (1) means 'good' pulses (CORRECT)
            else:
                break
                
        streamFile.close()
                   
    mlInput.m_classifier.fit(x_array_train, y_array_train)
        
    x_array_train.clear()
    y_array_train.clear()
    
    return mlInput.m_classifier.score(x_array_test, y_array_test), mlInput
      
"""

 This function determines the prediction accuracy of a TRAINed machine's classifier 'machineInput' by TESTing
 a different set of pulse streams containing correct and wrong pulses 'fileNameCorrectPulses' and 'fileNameRejectPulses' 
 repectively.
 
 If 'splitAfterNPulses' == -1 (default), the entire number of pulses from the given pulse streams 
 'fileNameCorrectPulses' and 'fileNameRejectPulses' are used for TESTing. Otherwise, the set number 
 'splitAfterNPulses' is considered for both pulse streams 'fileNameCorrectPulses' and 'fileNameRejectPulses'.
 
 return: 
     
     (1) prediction accuracy [0.0-1.0].
     
"""

def predictPulses(fileNameCorrectPulses = '/correct', 
                  fileNameRejectPulses  = '/reject', 
                  isPositivePolarity    = False,
                  splitAfterNPulses     = -1,
                  machineInput          = DMachineParams(),
                  debug                 = True):
    mlInput = machineInput.copy()
    
    fileSizeTrue  = os.path.getsize(fileNameCorrectPulses)
    fileSizeFalse = os.path.getsize(fileNameRejectPulses)
    
    readBytes = 32 #header offset
    
    x_array = []
    y_array = []
    
    # (1) REJECT pulses:
    with open(fileNameRejectPulses, "rb") as streamFile:
        if debug:
            print("(1) >> teach machine with REJECT pulses:\n")
            
        numberOfCells, sweepInNanoseconds, frequencyInGHz = readHeader(streamFile)
        
        pulseBytes = 2*numberOfCells*4
        numberOfPulses = (fileSizeFalse - readBytes)/pulseBytes
        
        if debug:
            print('number of cells:  {0}'.format(numberOfCells))
            print('sweep in ns:      {0}'.format(sweepInNanoseconds))
            print('frequency in GHz: {0}'.format(frequencyInGHz))
            print('number of pulses: {0}\n'.format(numberOfPulses))
        
        pulseCounter = 0
        
        while True:
            __, pulse = readPulse(streamFile, numberOfCells)
                
            if not len(pulse) or (splitAfterNPulses > -1 and pulseCounter > splitAfterNPulses):
                break
                
            # apply median filter?:
            if mlInput.m_medianFilter:
                pulse = medfilt(pulse, mlInput.m_windowSize)
                    
            # correct for baseline?:
            if mlInput.m_correctForBaseline:
                stddev_pre  = np.std(pulse[mlInput.m_startCell:mlInput.m_cellRegion])
                stddev_post = np.std(pulse[numberOfCells-1-mlInput.m_startCell-mlInput.m_cellRegion:numberOfCells-1-mlInput.m_startCell])
                    
                mean = 0.0
                    
                if np.abs(stddev_pre) < np.abs(stddev_post):
                    mean = np.mean(pulse[mlInput.m_startCell:mlInput.m_cellRegion])
                else:
                    mean = np.mean(pulse[numberOfCells-1-mlInput.m_startCell-mlInput.m_cellRegion:numberOfCells-1-mlInput.m_startCell])
                           
                pulse -= mean
                    
            readBytes += pulseBytes
            
            # normalize pulse data
            voltage_norm, __, __, valid = normalizeData(pulse, numberOfCells, isPositivePolarity)
                
            if not valid:
                continue
                
            pulseCounter += 1
            
            if debug:
                sys.stdout.write('\rread data: [{0}/{1}] MB <<>> [{2}/{3}] pulses]'.format((readBytes/1024)/1000, (fileSizeFalse/1024)/1000, (readBytes-32)/pulseBytes, numberOfPulses))
             
            x_array.append(voltage_norm)
            y_array.append(0) # << FALSE (0) means bad pulses (REJECT)
                
        streamFile.close()
        
    readBytes = 32 #header offset
    
    # (2) CORRECT pulses:
    with open(fileNameCorrectPulses, "rb") as streamFile:
        if debug:
            print("(2) >> teach machine with CORRECT pulses:\n")
            
        numberOfCells, sweepInNanoseconds, frequencyInGHz = readHeader(streamFile)
        
        pulseBytes = 2*numberOfCells*4
        numberOfPulses = (fileSizeTrue - readBytes)/pulseBytes
        
        if debug:
            print('number of cells:  {0}'.format(numberOfCells))
            print('sweep in ns:      {0}'.format(sweepInNanoseconds))
            print('frequency in GHz: {0}'.format(frequencyInGHz))
            print('number of pulses: {0}\n'.format(numberOfPulses))
        
        pulseCounter = 0
        
        while True:
            __, pulse = readPulse(streamFile, numberOfCells)
                
            if not len(pulse) or (splitAfterNPulses > -1 and pulseCounter > splitAfterNPulses):
                break
                
            # apply median filter?:
            if mlInput.m_medianFilter:
                pulse = medfilt(pulse, mlInput.m_windowSize)
                    
            # correct for baseline?:
            if mlInput.m_correctForBaseline:
                stddev_pre  = np.std(pulse[mlInput.m_startCell:mlInput.m_cellRegion])
                stddev_post = np.std(pulse[numberOfCells-1-mlInput.m_startCell-mlInput.m_cellRegion:numberOfCells-1-mlInput.m_startCell])
                    
                mean = 0.0
                    
                if np.abs(stddev_pre) < np.abs(stddev_post):
                    mean = np.mean(pulse[mlInput.m_startCell:mlInput.m_cellRegion])
                else:
                    mean = np.mean(pulse[numberOfCells-1-mlInput.m_startCell-mlInput.m_cellRegion:numberOfCells-1-mlInput.m_startCell])
                           
                pulse -= mean
                    
            readBytes += pulseBytes
            
            # normalize pulse data
            voltage_norm, __, __, valid = normalizeData(pulse, numberOfCells, isPositivePolarity)
                
            if not valid:
                continue
            
            pulseCounter += 1
            
            if debug:
                sys.stdout.write('\rread data: [{0}/{1}] MB <<>> [{2}/{3}] pulses]'.format((readBytes/1024)/1000, (fileSizeTrue/1024)/1000, (readBytes-32)/pulseBytes, numberOfPulses))
             
            x_array.append(voltage_norm)
            y_array.append(1) # << TRUE (1) means good pulses (CORRECT)
                
        streamFile.close()
        
    score = mlInput.m_classifier.score(x_array, y_array)
    
    if debug:
        sys.stdout.write('\nscore: {0}%'.format(score*100.0))
    
    return score

"""

 This function TRAINs the machine's classifier from the pulse streams containing the correct and wrong pulses 
 'fileNameCorrectPulses' and 'fileNameRejectPulses', respectively.
 
 If 'splitAfterNPulsesX' == -1 (default), the entire number of pulses from the given pulse streams 
 'fileNameCorrectPulses' and 'fileNameRejectPulses' are used for TRAINing. Otherwise, the set number 
 'splitAfterNPulsesX' is considered for both 'fileNameCorrectPulses' and 'fileNameRejectPulses'.
 
 If 'outputMachineFileName' == '', the TRAINed machine isn't stored in a file (*joblib). 
 
 return: 
     
     (1) TRAINed machine (DMachineParams()).
     
"""
    
def trainPulses(fileNameCorrectPulses    = '/correct', 
                fileNameRejectPulses     = '/reject', 
                outputMachineFileName    = '/machine', 
                isPositivePolarity       = False,
                splitAfterNPulsesCorrect = -1,
                splitAfterNPulsesReject  = -1,
                machineInput             = DMachineParams(),
                debug                    = True):
    mlInput = machineInput.copy()
    
    fileSizeTrue  = os.path.getsize(fileNameCorrectPulses)
    fileSizeFalse = os.path.getsize(fileNameRejectPulses)
    
    readBytes     = 32 #header offset
    
    x_array       = []
    y_array       = []
    
    # (1) REJECT pulses:
    with open(fileNameRejectPulses, "rb") as streamFile:
        if debug:
            print("(1) >> teach machine with REJECT pulses:\n")
            
        numberOfCells, sweepInNanoseconds, frequencyInGHz = readHeader(streamFile)
        
        pulseBytes = 2*numberOfCells*4
        numberOfPulses = (fileSizeFalse - readBytes)/pulseBytes
        
        if debug:
            print('number of cells:  {0}'.format(numberOfCells))
            print('sweep in ns:      {0}'.format(sweepInNanoseconds))
            print('frequency in GHz: {0}'.format(frequencyInGHz))
            print('number of pulses: {0}\n'.format(numberOfPulses))
        
        pulseCounter = 0
        
        while True:
            __, pulse = readPulse(streamFile, numberOfCells)
                
            if not len(pulse) or (splitAfterNPulsesReject > -1 and pulseCounter > splitAfterNPulsesReject):
                break
                
            # apply median filter?:
            if mlInput.m_medianFilter:
                pulse = medfilt(pulse, mlInput.m_windowSize)
                    
            # correct for baseline?:
            if mlInput.m_correctForBaseline:
                stddev_pre  = np.std(pulse[mlInput.m_startCell:mlInput.m_cellRegion])
                stddev_post = np.std(pulse[numberOfCells-1-mlInput.m_startCell-mlInput.m_cellRegion:numberOfCells-1-mlInput.m_startCell])
                    
                mean = 0.0
                    
                if np.abs(stddev_pre) < np.abs(stddev_post):
                    mean = np.mean(pulse[mlInput.m_startCell:mlInput.m_cellRegion])
                else:
                    mean = np.mean(pulse[numberOfCells-1-mlInput.m_startCell-mlInput.m_cellRegion:numberOfCells-1-mlInput.m_startCell])
                           
                pulse -= mean
                    
            readBytes += pulseBytes
            
            # normalize pulse data
            voltage_norm, __, __, valid = normalizeData(pulse, numberOfCells, isPositivePolarity)
                
            if not valid:
                continue
                
            pulseCounter += 1
            
            if debug:
                sys.stdout.write('\rread data: [{0}/{1}] MB <<>> [{2}/{3}] pulses]'.format((readBytes/1024)/1000, (fileSizeFalse/1024)/1000, (readBytes-32)/pulseBytes, numberOfPulses))
             
            x_array.append(voltage_norm)
            y_array.append(0) # << FALSE (0) means bad pulses (REJECT)
                
        streamFile.close()
        
    readBytes = 32 #header offset
    
    # (2) CORRECT pulses:
    with open(fileNameCorrectPulses, "rb") as streamFile:
        if debug:
            print("(2) >> teach machine with CORRECT pulses:\n")
            
        numberOfCells, sweepInNanoseconds, frequencyInGHz = readHeader(streamFile)
        
        pulseBytes = 2*numberOfCells*4
        numberOfPulses = (fileSizeTrue - readBytes)/pulseBytes
        
        if debug:
            print('number of cells:  {0}'.format(numberOfCells))
            print('sweep in ns:      {0}'.format(sweepInNanoseconds))
            print('frequency in GHz: {0}'.format(frequencyInGHz))
            print('number of pulses: {0}\n'.format(numberOfPulses))
        
        pulseCounter = 0
        
        while True:
            time, pulse = readPulse(streamFile, numberOfCells)
                
            if not len(pulse) or (splitAfterNPulsesCorrect > -1 and pulseCounter > splitAfterNPulsesCorrect):
                break
            
            # apply median filter?:
            if mlInput.m_medianFilter:
                pulse = medfilt(pulse, mlInput.m_windowSize)
                    
            # correct for baseline?:
            if mlInput.m_correctForBaseline:
                stddev_pre  = np.std(pulse[mlInput.m_startCell:mlInput.m_cellRegion])
                stddev_post = np.std(pulse[numberOfCells-1-mlInput.m_startCell-mlInput.m_cellRegion:numberOfCells-1-mlInput.m_startCell])
                    
                mean = 0.0
                    
                if np.abs(stddev_pre) < np.abs(stddev_post):
                    mean = np.mean(pulse[mlInput.m_startCell:mlInput.m_cellRegion])
                else:
                    mean = np.mean(pulse[numberOfCells-1-mlInput.m_startCell-mlInput.m_cellRegion:numberOfCells-1-mlInput.m_startCell])
                           
                pulse -= mean
                
            readBytes += pulseBytes
            
            # normalize pulse data
            voltage_norm, __, __, valid = normalizeData(pulse, numberOfCells, isPositivePolarity)
            
            if not valid:
                continue
            
            pulseCounter += 1
            
            if debug:
                sys.stdout.write('\rread data: [{0}/{1}] MB <<>> [{2}/{3}] pulses]'.format((readBytes/1024)/1000, (fileSizeTrue/1024)/1000, (readBytes-32)/pulseBytes, numberOfPulses))
             
            x_array.append(voltage_norm)
            y_array.append(1) # << TRUE (1) means good pulses (CORRECT)
                
        streamFile.close()
                
    mlInput.m_classifier.fit(x_array, y_array)
            
    x_array.clear()
    y_array.clear()
        
    if not outputMachineFileName == '':
        mlInput.save(outputMachineFileName)
        
    return mlInput
        
"""

 This function is similar to 'trainPulses(..)' but, instead, applies batching (out-of-core fitting) internally 
 to TRAIN the machine's classifier on large pulse streams, which do not fit the RAM.
 
 If 'splitAfterNPulses' == -1 (default), the entire number pulses from the given pulse streams 
 'fileNameCorrectPulses' and 'fileNameRejectPulses' are used to TRAIN the machine's classifier. Otherwise, the set number 
 'splitAfterNPulses' is considered for both pulse streams 'fileNameCorrectPulses' and 'fileNameRejectPulses'.
 
 If 'outputMachineFileName' == '', the machine won't be stored in a file (*joblib). 
 
 return: 
     
     (1) TRAINed machine (DMachineParams()).
     
"""

def trainPulsesOnline(fileNameCorrectPulses = '/correct', 
                      fileNameRejectPulses  = '/reject', 
                      outputMachineFileName = '/machine', 
                      isPositivePolarity    = False,
                      chunkSize             = 5000, # according to the specs of scikit learn, set this number as near as possible to the RAM size.
                      machineInput          = DMachineParams(),
                      debug                 = True): 
    mlInput = machineInput.copy()
    
    y_all = []
    y_all.append(0) # bad  pulse (REJECT)
    y_all.append(1) # good pulse (CORRECT)
    
    fileSizeTrue  = os.path.getsize(fileNameCorrectPulses)
    fileSizeFalse = os.path.getsize(fileNameRejectPulses)
    
    readBytes = 32 #header offset
    
    # (1) REJECT pulses:
    with open(fileNameRejectPulses, "rb") as streamFile:
        if debug:
            print("(1) >> teach machine with REJECT pulses:\n")
            
        numberOfCells, sweepInNanoseconds, frequencyInGHz = readHeader(streamFile)
        
        pulseBytes     = 2*numberOfCells*4
        numberOfPulses = (fileSizeFalse - readBytes)/pulseBytes
        
        if debug:
            print('number of cells:  {0}'.format(numberOfCells))
            print('sweep in ns:      {0}'.format(sweepInNanoseconds))
            print('frequency in GHz: {0}'.format(frequencyInGHz))
            print('number of pulses: {0}\n'.format(numberOfPulses))
        
        pulseCounter = 0
        
        while True:
            x_array = []
            y_array = []
           
            abortStream = False
            
            for run in range(0, chunkSize):
                __, pulse = readPulse(streamFile, numberOfCells)
                
                if not len(pulse):
                    abortStream = True
                    break
                
                # apply median filter?:
                if mlInput.m_medianFilter:
                    pulse = medfilt(pulse, mlInput.m_windowSize)
                    
                # correct for baseline?:
                if mlInput.m_correctForBaseline:
                    mean_pre  = np.mean(pulse[mlInput.m_startCell:mlInput.m_cellRegion])
                    mean_post = np.mean(pulse[numberOfCells-1-mlInput.m_startCell-mlInput.m_cellRegion:numberOfCells-1-mlInput.m_startCell])
                    
                    stddev_pre  = np.std(pulse[mlInput.m_startCell:mlInput.m_cellRegion])
                    stddev_post = np.std(pulse[numberOfCells-1-mlInput.m_startCell-mlInput.m_cellRegion:numberOfCells-1-mlInput.m_startCell])
                    
                    mean = 0.0
                    
                    if np.abs(stddev_pre) < np.abs(stddev_post):
                        mean = mean_pre
                    else:
                        mean = mean_post
                           
                    pulse -= mean
                    
                readBytes += pulseBytes
                
                voltage_norm, __, __, valid = normalizeData(pulse, numberOfCells, isPositivePolarity)
                
                if not valid:
                    continue
                
                pulseCounter += 1
                
                if debug:
                    sys.stdout.write('\rread data: [{0}/{1}] MB <<>> [{2}/{3}] pulses]'.format((readBytes/1024)/1000, (fileSizeFalse/1024)/1000, (readBytes-32)/pulseBytes, numberOfPulses))
             
                x_array.append(voltage_norm)
                y_array.append(0)
                
            if len(x_array) > 0:    
                mlInput.m_classifier.partial_fit(x_array, y_array, classes=np.unique(y_all))
                
            if abortStream:
                break;
                
        streamFile.close()
        
    readBytes = 32 #header offset   
        
    # (2) CORRECT pulses:
    with open(fileNameCorrectPulses, "rb") as streamFile2:
        if debug:
            print("(2) >> teach machine with CORRECT pulses:\n")
            
        numberOfCells, sweepInNanoseconds, frequencyInGHz = readHeader(streamFile2)
        
        pulseBytes = 2*numberOfCells*4
        numberOfPulses = (fileSizeTrue - readBytes)/pulseBytes
        
        if debug:
            print('number of cells:  {0}'.format(numberOfCells))
            print('sweep in ns:      {0}'.format(sweepInNanoseconds))
            print('frequency in GHz: {0}'.format(frequencyInGHz))
            print('number of pulses: {0}\n'.format(numberOfPulses))
            
        pulseCounter = 0
        
        while True:
            x_array = []
            y_array = []
           
            abortStream = False
            
            for run in range(0, chunkSize):
                __, pulse = readPulse(streamFile2, numberOfCells)
                
                if not len(pulse):
                    abortStream = True
                    break
                
                # apply median filter?:
                if mlInput.m_medianFilter:
                    pulse = medfilt(pulse, mlInput.m_windowSize)
                    
                # correct for baseline?:
                if mlInput.m_correctForBaseline:
                    mean_pre  = np.mean(pulse[mlInput.m_startCell:mlInput.m_cellRegion])
                    mean_post = np.mean(pulse[numberOfCells-1-mlInput.m_startCell-mlInput.m_cellRegion:numberOfCells-1-mlInput.m_startCell])
                    
                    mean = 0.0
                    
                    if np.abs(mean_pre) < np.abs(mean_post):
                        mean = mean_pre
                    else:
                        mean = mean_post
                           
                    pulse -= mean
                
                
                readBytes += pulseBytes
                
                voltage_norm, __, __, valid = normalizeData(pulse, numberOfCells, isPositivePolarity)
                
                if not valid:
                    continue
                
                pulseCounter += 1
                
                if debug:
                    sys.stdout.write('\rread data: [{0}/{1}] MB <<>> [{2}/{3}] pulses]'.format((readBytes/1024)/1000, (fileSizeTrue/1024)/1000, (readBytes-32)/pulseBytes, numberOfPulses))
            
                x_array.append(voltage_norm)
                y_array.append(1)
                
            if len(x_array) > 0:    
                mlInput.m_classifier.partial_fit(x_array, y_array, classes=np.unique(y_all))
                
            if abortStream:
                break;
                
        streamFile2.close()
    
    if not outputMachineFileName == '':
        mlInput.save(outputMachineFileName)
    
    return mlInput
     
"""

 This function can be used to TRAIN and TEST a machine's classifier from TWO data sets of correct ('fileNameCorrectPulses_train' & 'fileNameCorrectPulses_test') 
 and wrong ('fileNameRejectPulses_train' & 'fileNameRejectPulses_test') streamed pulses.
 
 This function is similar to 'splitTrainAndTest(..)', which instead needs only ONE data set of pulse streams.

 If 'splitAfterNPulses_xx' == -1 (default), the entire number pulses from the pulse streams 
 'fileNameCorrectPulses_xx' and 'fileNameRejectPulses_xx' are used for TRAINing or TESTing. Otherwise, the given number 
 'splitAfterNPulses_xx' is considered for both pulse streams 'fileNameCorrectPulses_xx' and 'fileNameRejectPulses_xx'.
 
 return: 
     
     (1) list, which contains the prediction accuracies [0.0-1.0] for each 'numberOfPulses_train',
     (4) DMachineParams() from the learned machine.
 
"""

def trainAndTest(fileNameCorrectPulses_train = '/correct_train', 
                 fileNameRejectPulses_train  = '/reject_train',
                 fileNameCorrectPulses_test  = '/correct_test', 
                 fileNameRejectPulses_test   = '/reject_test',
                 numberOfPulsesCorrect_train = -1,
                 numberOfPulsesReject_train  = -1,
                 numberOfPulses_test         = -1,
                 isPositivePolarity          = False,
                 machineInput                = DMachineParams()):
    mlInput = machineInput.copy()
    
    # train
    learnedMachine = trainPulses(fileNameCorrectPulses_train, 
                                 fileNameRejectPulses_train, 
                                 '', 
                                 isPositivePolarity, 
                                 numberOfPulsesCorrect_train,
                                 numberOfPulsesReject_train,
                                 mlInput, 
                                 False)
    
    # test
    score = predictPulses(fileNameCorrectPulses_test, 
                          fileNameRejectPulses_test, 
                          isPositivePolarity,  
                          numberOfPulses_test,
                          learnedMachine,
                          False)
    
    return score, learnedMachine

"""

 This function executes a pipeline on a TESTing ('_test') and TRAINing ('_train') set of 
 streamed pulses ('fileNameCorrectPulses_xx' and 'fileNameRejectPulses_xx') by varying the 
 number of TRAINed pulses for a given machine definition (classifier) 'machineInput'.

 The list 'numberOfPulses_train' defines the region of N pulses used for the TRAINing cycle, f.e.: 
     
 [2, 50, 2] means: looping from 2 to (=)50 with an increment of 2. 
 
 The number of TRAINed pulses 'numberOfPulses_train' is equally applied for CORRECT and REJECT pulses.

 return: 
     
     (1) list, which contains x-axis data: 'numberOfPulses_train',
     (2) list, which contains the prediction accuracies [0.0-1.0] for each 'numberOfPulses_train' (1).
 
"""

def runPipelineNPulses(fileNameCorrectPulses_train = '/correct_train', 
                       fileNameRejectPulses_train  = '/reject_train', 
                       fileNameCorrectPulses_test  = '/correct_test', 
                       fileNameRejectPulses_test   = '/correct_test', 
                       numberOfPulses_train        = [2, 50, 1],
                       numberOfPulses_test         = 1000,
                       isPositivePolarity          = False,
                       machineInput                = DMachineParams(),
                       debug                       = True):
   mlInput = machineInput.copy()
   
   plArrX   = []
   plArrY   = []
   
   size = (numberOfPulses_train[1] + numberOfPulses_train[2])/numberOfPulses_train[2]
   counter = 0
    
   for N in range(numberOfPulses_train[0], numberOfPulses_train[1] + numberOfPulses_train[2], numberOfPulses_train[2]): 
       counter += 1
       
       score, __   = trainAndTest(fileNameCorrectPulses_train, 
                                  fileNameRejectPulses_train,
                                  fileNameCorrectPulses_test, 
                                  fileNameRejectPulses_test,
                                  N,
                                  N,
                                  numberOfPulses_test,
                                  isPositivePolarity,
                                  mlInput)
       plArrX. append(N)
       plArrY. append(score)
       
       if debug:
           sys.stdout.write('\rprogress: [{0}/{1}] = {2}%'.format(counter, size, 100.0*counter/size))
       
   return plArrX, plArrY

"""

 This function executes a pipeline on a TESTing ('_test') and TRAINing ('_train') set of 
 streamed pulses ('fileNameCorrectPulses_xx' and 'fileNameRejectPulses_xx') by varying the 
 window size of the applied median filter (note: odd numbers are required!) for a given machine 
 definition (classifier) 'machineInput'.

 The list 'medianFilterIncr' defines the window size region used for the TRAINing cycle, f.e.: 
     
 [3, 31, 2] means: looping from 3 to (=)31 with an increment of 2 to consider ODD numbers.
 
 Note: the median filter requires ODD window sizes for each step in the pipeline.

 The number of TRAINed pulses 'numberOfPulses_train' is equally applied for CORRECT and REJECT pulses.
 
 return: 
     
     (1) list, which contains x-axis data: 'medianFilterIncr',
     (2) list, which contains the prediction accuracies [0.0-1.0] for each 'medianFilterIncr' (1).
 
"""

def runPipelineMedianWindow(fileNameCorrectPulses_train = '/correct_train', 
                            fileNameRejectPulses_train  = '/reject_train', 
                            fileNameCorrectPulses_test  = '/correct_test', 
                            fileNameRejectPulses_test   = '/correct_test', 
                            numberOfPulses_train        = 15,
                            numberOfPulses_test         = 1000,
                            isPositivePolarity          = False,
                            machineInput                = DMachineParams(),
                            medianFilterIncr            = [3, 31, 2],
                            debug                       = True):
   mlInput = machineInput.copy()
    
   plArrX   = []
   plArrY   = []
   
   size = (medianFilterIncr[1] + medianFilterIncr[2])/medianFilterIncr[2]
   counter = 0
   
   for N in range(medianFilterIncr[0], medianFilterIncr[1] + medianFilterIncr[2], medianFilterIncr[2]): 
       mlInput.m_windowSize = N 
       
       counter += 1
       
       score, __   = trainAndTest(fileNameCorrectPulses_train, 
                                  fileNameRejectPulses_train,
                                  fileNameCorrectPulses_test, 
                                  fileNameRejectPulses_test,
                                  numberOfPulses_train,
                                  numberOfPulses_train,
                                  numberOfPulses_test,
                                  isPositivePolarity,
                                  mlInput)
       plArrX. append(N)
       plArrY. append(score)
       
       if debug:
           sys.stdout.write('\rprogress: [{0}/{1}] = {2}%'.format(counter, size, 100.0*counter/size))
            
   return plArrX, plArrY

"""

 This function combines functions: runPipelineNPulses(..) and runPipelineMedianWindow(..).
 
"""

def runPipelineGrid(fileNameCorrectPulses_train = '/correct_train', 
                    fileNameRejectPulses_train  = '/reject_train', 
                    fileNameCorrectPulses_test  = '/correct_test', 
                    fileNameRejectPulses_test   = '/correct_test', 
                    numberOfPulses_train        = [2, 50, 1],
                    medianFilterIncr            = [3, 31, 2],
                    numberOfPulses_test         = 1000,
                    isPositivePolarity          = False,
                    machineInput                = DMachineParams(),
                    debug                       = True):
    mlInput = machineInput.copy()
    
    _plArrY   = []
   
    _xAxis    = np.arange(numberOfPulses_train[0], numberOfPulses_train[1] + numberOfPulses_train[2], numberOfPulses_train[2]) # number of pulses (N CORRECT = N REJECT)
    _yAxis    = np.arange(medianFilterIncr[0],     medianFilterIncr[1] + medianFilterIncr[2],         medianFilterIncr[2])     # median filter window size
    
    counter = 0
    
    for N_m in range(medianFilterIncr[0], medianFilterIncr[1] + medianFilterIncr[2], medianFilterIncr[2]):
        mlInput.m_windowSize = N_m
        
        plArrY   = []
        
        for N_p in range(numberOfPulses_train[0], numberOfPulses_train[1] + numberOfPulses_train[2], numberOfPulses_train[2]): 
           counter += 1
            
           score, __   = trainAndTest(fileNameCorrectPulses_train, 
                                      fileNameRejectPulses_train,
                                      fileNameCorrectPulses_test, 
                                      fileNameRejectPulses_test,
                                      N_p,
                                      N_p,
                                      numberOfPulses_test,
                                      isPositivePolarity,
                                      mlInput)
           
           if debug:
               sys.stdout.write('\rprogress: [{0}/{1}] = {2}%'.format(counter, len(_xAxis)*len(_yAxis), 100.0*counter/(len(_xAxis)*len(_yAxis))))
            
           plArrY.append(score)
           
        _plArrY. append(plArrY)
        
    return _xAxis, _yAxis, _plArrY

"""

 This function executes a pipeline on a TESTing ('_test') and TRAINing ('_train') set of 
 streamed pulses ('fileNameCorrectPulses_xx' and 'fileNameRejectPulses_xx') by varying the 
 number of CORRECT pulses vs. the number of REJECT pulses for a given machine 
 definition (classifier) 'machineInput'.

 The list 'numberOfPulsesCorrect_xxx' defines the number of pulses used for the TRAINing cycle, f.e.: 
     
 [3, 31, 2] means: looping from 3 to (=)31 with an increment of 2.
 
 return: 
     
     (1) list, which contains x-axis data: 'numberOfPulsesCorrect_train',
     (2) list, which contains y-axis data: 'fileNameRejectPulses_train',
     (3) list, which contains the prediction accuracies [0.0-1.0] for each point (x,y).
     (4) point (x,y) of the best prediction accuracy and the repsective accuracy score [0.0-1.0].
 
"""

def runPipelineGrid2(fileNameCorrectPulses_train = '/correct_train', 
                     fileNameRejectPulses_train  = '/reject_train', 
                     fileNameCorrectPulses_test  = '/correct_test', 
                     fileNameRejectPulses_test   = '/correct_test', 
                     numberOfPulsesCorrect_train = [2, 50, 1],
                     numberOfPulsesReject_train  = [2, 50, 1],
                     numberOfPulses_test         = 1000,
                     isPositivePolarity          = False,
                     machineInput                = DMachineParams(),
                     debug                       = True):
    mlInput = machineInput.copy()
    
    _plArrY   = []
   
    _xAxis    = np.arange(numberOfPulsesCorrect_train[0], numberOfPulsesCorrect_train[1] + numberOfPulsesCorrect_train[2], numberOfPulsesCorrect_train[2]) # number of pulses: CORRECT
    _yAxis    = np.arange(numberOfPulsesReject_train[0],  numberOfPulsesReject_train[1] + numberOfPulsesReject_train[2],   numberOfPulsesReject_train[2])  # number of pulses: REJECT
    
    bestList  = [0.0, 0, 0]
    maxScore  = -1.0       

    counter = 0              
                         
    for N_r in range(numberOfPulsesReject_train[0], numberOfPulsesReject_train[1] + numberOfPulsesReject_train[2], numberOfPulsesReject_train[2]):
        plArrY   = []
        
        for N_c in range(numberOfPulsesCorrect_train[0], numberOfPulsesCorrect_train[1] + numberOfPulsesCorrect_train[2], numberOfPulsesCorrect_train[2]): 
           counter += 1
           
           score, __   = trainAndTest(fileNameCorrectPulses_train, 
                                      fileNameRejectPulses_train,
                                      fileNameCorrectPulses_test, 
                                      fileNameRejectPulses_test,
                                      N_c,
                                      N_r,
                                      numberOfPulses_test,
                                      isPositivePolarity,
                                      mlInput)
           
           if debug:
               sys.stdout.write('\rprogress: [{0}/{1}] = {2}%'.format(counter, len(_xAxis)*len(_yAxis), 100.0*counter/(len(_xAxis)*len(_yAxis))))
             
           if score >= maxScore:
               maxScore    = score
               
               bestList[0] = score
               bestList[1] = N_c
               bestList[2] = N_r
           
           plArrY.append(score)
           
        _plArrY. append(plArrY)
        
    return _xAxis, _yAxis, _plArrY, bestList

"""

 This function creates a lifetime spectrum from a sample pulse stream 'pulseStreamFile'
 using the TRAINned/learned machines 'machineInputA' and 'machineInputB' for detector A and B respectively. 
 The resulting lifetime spectrum is stored in the file 'outputName' after each 100 counts acquired.
 
 params: 
     
   'isPositivePolarity'                   >> pulse polarity: 'True' for (+)  and 'False' for (-)
   binWidth_in_ps                         >> bin width (channel width) in picoseconds
   numberOfBins                           >> number of bins (channels) used to define the lifetime spectrum
   offset_in_ps                           >> offset, which appr. defines the region of the peak maximum in the lifetime spectrum
   B_as_start_A_as_stop                   >> if 'TRUE': B = start and A = stop, if 'False': B = stop and A = start
   cf_level_A                             >> CF level in percentage (%) for detector A
   cf_level_B                             >> CF level in percentage (%) for detector B
   ll_phs_start_in_mV, ul_phs_start_in_mV >> lower/upper level in absolute values of millivolts [mV] for the accepted pulse heights (amplitudes) of the start branch
   ll_phs_stop_in_mV,  ul_phs_stop_in_mV  >> lower/upper level in absolute values of millivolts [mV] for the accepted pulse heights (amplitudes) of the stop branch
   cubicSpline                            >> if 'True', a cubic spline with a render depth of 'cubicSplineRenderPoints' between two neighbouring sampling points is applied for the determination of the CF level,
   cubicSplineRenderPoints                >> see 'cubicSpline'
   medianFilterA                          >> if 'True', a median filter is applied with the given window size 'windowSizeA' on the pulse data of detector A
   windowSizeA                            >> see 'medianFilterA'
   medianFilterB                          >> if 'True', a median filter is applied with the given window size 'windowSizeB' on the pulse data of detector B
   windowSizeB                            >> see 'medianFilterB'
 
"""

def createLifetimeSpectrum(machineInputA           = DMachineParams(),
                           machineInputB           = DMachineParams(),
                           pulseStreamFile         = '/pulsePairStream', 
                           outputName              = '/spectrum',
                           isPositivePolarity      = False,
                           binWidth_in_ps          = 5,
                           numberOfBins            = 28000,
                           offset_in_ps            = 0.0,
                           B_as_start_A_as_stop    = True,
                           cf_level_A              = 25.0,
                           cf_level_B              = 25.0,
                           ll_phs_start_in_mV      = 250.0, ul_phs_start_in_mV = 450.0,
                           ll_phs_stop_in_mV       = 50.0,  ul_phs_stop_in_mV  = 150.0,
                           cubicSpline             = True,
                           cubicSplineRenderPoints = 200,
                           medianFilterA           = True,
                           windowSizeA             = 5,
                           medianFilterB           = True,
                           windowSizeB             = 5,
                           debug                   = True):
    # (1) retrieve classifier:
    classifierA = machineInputA.m_classifier
    classifierB = machineInputB.m_classifier
    
    # (2) open pulse stream and read header to extract necessary information:
    fileSize = os.path.getsize(pulseStreamFile)
        
    with open(pulseStreamFile, "rb") as streamFile:
        numberOfCells, sweepInNanoseconds, frequencyInGHz = readHeader(streamFile)
        
        if debug:
            print('number of cells:  {0}'.format(numberOfCells))
            print('sweep in ns:      {0}'.format(sweepInNanoseconds))
            print('frequency in GHz: {0}\n'.format(frequencyInGHz))
        
        readBytes  = 32                #header offset
        pulseBytes = 4*numberOfCells*4 #pulse pair size
        
        lifetimeSpectrum        = np.zeros(numberOfBins)
        overall_region_in_ps    = numberOfBins*binWidth_in_ps
        
        countsInSpectrum        = 0

        while True:
            x_arrayA = []
            x_arrayB = []
            
            timeA, pulseA = readPulse(streamFile, numberOfCells)
            
            if not len(pulseA):
                break
            
            timeB, pulseB = readPulse(streamFile, numberOfCells)
            
            if not len(pulseB):
                break
            
            readBytes += pulseBytes
            
            # hold copy of original pulses
            pulseA_origin = np.zeros(numberOfCells)
            pulseB_origin = np.zeros(numberOfCells)
            
            pulseA_origin[:] = pulseA
            pulseB_origin[:] = pulseB
            
            # apply median filter on ML data?:
            if machineInputA.m_medianFilter:
                pulseA = medfilt(pulseA, machineInputA.m_windowSize)
                
            if machineInputB.m_medianFilter:
                pulseB = medfilt(pulseB, machineInputB.m_windowSize)
                
            # apply median filter on original data?:
            if medianFilterA:
                pulseA_origin = medfilt(pulseA_origin, windowSizeA)
                
            if medianFilterB:
                pulseB_origin = medfilt(pulseB_origin, windowSizeB)
                
            # correct for baseline?:
            if machineInputA.m_correctForBaseline:
                meanA_pre      = np.mean(pulseA[machineInputA.m_startCell:machineInputA.m_cellRegion])
                meanA_post     = np.mean(pulseA[numberOfCells-1-machineInputA.m_startCell-machineInputA.m_cellRegion:numberOfCells-1-machineInputA.m_startCell])
                
                stddevA_pre    = np.std(pulseA[machineInputA.m_startCell:machineInputA.m_cellRegion])
                stddevA_post   = np.std(pulseA[numberOfCells-1-machineInputA.m_startCell-machineInputA.m_cellRegion:numberOfCells-1-machineInputA.m_startCell])
                
                meanA_pre_o    = np.mean(pulseA_origin[machineInputA.m_startCell:machineInputA.m_cellRegion])
                meanA_post_o   = np.mean(pulseA_origin[numberOfCells-1-machineInputA.m_startCell-machineInputA.m_cellRegion:numberOfCells-1-machineInputA.m_startCell])
                
                stddevA_pre_o  = np.std(pulseA_origin[machineInputA.m_startCell:machineInputA.m_cellRegion])
                stddevA_post_o = np.std(pulseA_origin[numberOfCells-1-machineInputA.m_startCell-machineInputA.m_cellRegion:numberOfCells-1-machineInputA.m_startCell])
                 
                meanA   = 0.0
                meanA_o = 0.0
                
                if np.abs(stddevA_pre) < np.abs(stddevA_post):
                    meanA = meanA_pre
                else:
                    meanA = meanA_post
                    
                if np.abs(stddevA_pre_o) < np.abs(stddevA_post_o):
                    meanA_o = meanA_pre_o
                else:
                    meanA_o = meanA_post_o
                    
                pulseA        -= meanA
                pulseA_origin -= meanA_o
                                
            if machineInputB.m_correctForBaseline:
                meanB_pre      = np.mean(pulseB[machineInputB.m_startCell:machineInputB.m_cellRegion])
                meanB_post     = np.mean(pulseB[numberOfCells-1-machineInputB.m_startCell-machineInputB.m_cellRegion:numberOfCells-1-machineInputB.m_startCell])
                
                stddevB_pre    = np.std(pulseB[machineInputB.m_startCell:machineInputB.m_cellRegion])
                stddevB_post   = np.std(pulseB[numberOfCells-1-machineInputB.m_startCell-machineInputB.m_cellRegion:numberOfCells-1-machineInputB.m_startCell])
                
                meanB_pre_o    = np.mean(pulseB_origin[machineInputB.m_startCell:machineInputB.m_cellRegion])
                meanB_post_o   = np.mean(pulseB_origin[numberOfCells-1-machineInputB.m_startCell-machineInputB.m_cellRegion:numberOfCells-1-machineInputB.m_startCell])
                
                stddevB_pre_o  = np.std(pulseB_origin[machineInputB.m_startCell:machineInputB.m_cellRegion])
                stddevB_post_o = np.std(pulseB_origin[numberOfCells-1-machineInputB.m_startCell-machineInputB.m_cellRegion:numberOfCells-1-machineInputB.m_startCell])
                
                meanB   = 0.0
                meanB_o = 0.0
                
                if np.abs(stddevB_pre) < np.abs(stddevB_post):
                    meanB = meanB_pre
                else:
                    meanB = meanB_post
                    
                if np.abs(stddevB_pre_o) < np.abs(stddevB_post_o):
                    meanB_o = meanB_pre_o
                else:
                    meanB_o = meanB_post_o
                    
                pulseB        -= meanB
                pulseB_origin -= meanB_o
                
            # determine pulse height for original data
            if not isPositivePolarity:
                amplitudeA_o = np.min(pulseA_origin)
                amplitudeB_o = np.min(pulseB_origin)
            else:
                amplitudeA_o = np.max(pulseA_origin)
                amplitudeB_o = np.max(pulseB_origin)
                
            _pulseA = np.zeros(numberOfCells)
            _pulseB = np.zeros(numberOfCells)
            
            _pulseA[:] = pulseA
            _pulseB[:] = pulseB
            
            voltage_normA, __, __, validA = normalizeData(_pulseA, numberOfCells, isPositivePolarity)
            voltage_normB, __, __, validB = normalizeData(_pulseB, numberOfCells, isPositivePolarity)
            
            if not validA or not validB:
                continue
            
            x_arrayA.append(voltage_normA)
            resultA = classifierA.predict(x_arrayA)
            
            x_arrayB.append(voltage_normB)
            resultB = classifierB.predict(x_arrayB)
            
            if not (resultA[0] == 1 and resultB[0] == 1):
                continue
            
            
            __amplitudeA = np.abs(amplitudeA_o)
            __amplitudeB = np.abs(amplitudeB_o)
                
            acceptForLTSpec = False
            
            #plt.plot(pulseA_origin, 'ro', pulseB_origin, 'bo')
            #plt.show()
            
            if B_as_start_A_as_stop:
                acceptForLTSpec = (__amplitudeB >= ll_phs_start_in_mV and __amplitudeB <= ul_phs_start_in_mV) and (__amplitudeA >= ll_phs_stop_in_mV and __amplitudeA <= ul_phs_stop_in_mV)
            else:
                acceptForLTSpec = (__amplitudeA >= ll_phs_start_in_mV and __amplitudeA <= ul_phs_start_in_mV) and (__amplitudeB >= ll_phs_stop_in_mV and __amplitudeB <= ul_phs_stop_in_mV)
                    
            # (3) calculate lifetime
            lifetime_in_ps = 0.0
            rejectLT       = True

            if acceptForLTSpec and B_as_start_A_as_stop:
                lifetime_in_ps, rejectLT = calcLifetime(timeB, pulseB_origin, timeA, pulseA_origin, cf_level_B, cf_level_A, amplitudeB_o, amplitudeA_o, isPositivePolarity, cubicSpline, cubicSplineRenderPoints)
            elif acceptForLTSpec and not B_as_start_A_as_stop:
                lifetime_in_ps, rejectLT = calcLifetime(timeA, pulseA_origin, timeB, pulseB_origin, cf_level_A, cf_level_B, amplitudeA_o, amplitudeB_o, isPositivePolarity, cubicSpline, cubicSplineRenderPoints)
                    
            # (4) bin lifetimes
            if acceptForLTSpec and not rejectLT:
                lifetime_in_ps += offset_in_ps
                index = (int)(((lifetime_in_ps/overall_region_in_ps)*numberOfBins)-1)
                
                if index >= 0 and index < numberOfBins:
                    lifetimeSpectrum[index] += 1
                    countsInSpectrum        += 1
                    
                    rb = (readBytes/1024)/1000
                    fs = (fileSize/1024)/1000
                    pe = 100.0*(rb/fs)
                    es = (fs*countsInSpectrum/rb)/1000000
                                         
                    if debug:
                        sys.stdout.write('\rbytes read: [{0}/{1}] MB ({2} %) >> integral counts: {3} << est. counts in spectrum: {4} Mio.'.format(rb, fs, pe, countsInSpectrum, es))
                        
                    # outsave
                    if not countsInSpectrum % 100: 
                        np.savetxt(outputName + '.txt', lifetimeSpectrum, fmt='%0d', newline='\n', header='counts [#]\n');
                            
                    # plot
                    if not countsInSpectrum % 100000:
                        plt.semilogy(lifetimeSpectrum,'ro')
                        plt.show()
                        
        streamFile.close()
        np.savetxt(outputName, lifetimeSpectrum, fmt='%0d', newline='\n', header='counts [#]\n')
