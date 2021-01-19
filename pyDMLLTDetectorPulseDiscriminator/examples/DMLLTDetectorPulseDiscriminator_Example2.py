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

import matplotlib.pyplot as plt
import pylab as pl
import numpy as np

from DMLLTDetectorPulseDiscriminator import *

"""

This example demonstrates how to TRAIN your machine*s classifiers for detector A and B and 
subsequently generate the lifetime spectrum.
 
"""

########################## DEFINITIONS ##################################
#
relPath            = 'C:/Users/danny/Desktop/__AILearning/'

"""
 define your TRAINing data set for detector A and B:
"""
filenameTrue_A     = relPath + 'A/true/A.drs4DataStream'
filenameFalse_A    = relPath + 'A/false/A.drs4DataStream'

filenameTrue_B     = relPath + 'B/true/B.drs4DataStream'
filenameFalse_B    = relPath + 'B/false/B.drs4DataStream'  

"""
 pulse stream acquired from the studied sample, which is used to generate the lifetime spectrum:
"""
dataStream         = 'E:/Fe/Fe_14_09_2018.drs4DataStream'

"""
 location and filename where the resulting lifetime spectrum will be stored:
"""
spectraOutput      = relPath + 'spectrum/spectrum_Fe'
#
#########################################################################

"""
 configure machine params of pulse (A):
"""
mlinputA = DMachineParams()

# baseline correction
mlinputA.m_startCell  = 10
mlinputA.m_cellRegion = 150

# median filter
mlinputA.m_medianFilter = True
mlinputA.m_windowSize   = 5

mlinputA.m_classifier = CalibratedClassifierCV(GaussianNB(), cv=2, method='isotonic')

"""
 configure machine params of pulse (B):
"""
mlinputB = DMachineParams()

# baseline correction
mlinputB.m_startCell  = 10
mlinputB.m_cellRegion = 150

# median filter
mlinputB.m_medianFilter = True
mlinputB.m_windowSize   = 5

mlinputB.m_classifier = CalibratedClassifierCV(GaussianNB(), cv=2, method='isotonic')


"""
 define the number of correct/reject detector pulses (A/B), which should be used to TRAIN the machine's classifier prior to the lifetime spectrum generation:
"""
numberOfPulsesToBeTrainedCorrect_A = 16
numberOfPulsesToBeTrainedReject_A  = 14

numberOfPulsesToBeTrainedCorrect_B = 14
numberOfPulsesToBeTrainedReject_B  = 18

"""
 train the machine's classifier for detector (A):
"""
mlinputA = trainPulses(fileNameCorrectPulses    = filenameTrue_A, 
                       fileNameRejectPulses     = filenameFalse_A, 
                       outputMachineFileName    = '', 
                       isPositivePolarity       = False,
                       splitAfterNPulsesCorrect = numberOfPulsesToBeTrainedCorrect_A,
                       splitAfterNPulsesReject  = numberOfPulsesToBeTrainedReject_A,
                       machineInput             = mlinputA,
                       debug                    = True)

"""
 train the machine's classifier for detector (B):
"""
mlinputB = trainPulses(fileNameCorrectPulses    = filenameTrue_B, 
                       fileNameRejectPulses     = filenameFalse_B, 
                       outputMachineFileName    = '', 
                       isPositivePolarity       = False,
                       splitAfterNPulsesCorrect = numberOfPulsesToBeTrainedCorrect_B,
                       splitAfterNPulsesReject  = numberOfPulsesToBeTrainedReject_B,
                       machineInput             = mlinputB,
                       debug                    = True)

"""
 generate the lifetime spectrum:
"""
createLifetimeSpectrum(machineInputA           = mlinputA,
                       machineInputB           = mlinputB,
                       pulseStreamFile         = dataStream, 
                       outputName              = spectraOutput,
                       
                       # negative pulse polarity
                       isPositivePolarity      = False,
                       
                       binWidth_in_ps          = 5,
                       numberOfBins            = 28000,   # [ps]
                       offset_in_ps            = 17000.0, # [ps]
                       B_as_start_A_as_stop    = True,
                       
                       # constant fraction (CF) levels
                       cf_level_A              = 25.0,    # [%]
                       cf_level_B              = 25.0,    # [%]
                       
                       # PHS (start)
                       ll_phs_start_in_mV      = 259.0,   # [mV]
                       ul_phs_start_in_mV      = 319.0,   # [mV]
                       
                       # PHS (stop)
                       ll_phs_stop_in_mV       = 67.0,    # [mV]
                       ul_phs_stop_in_mV       = 107.0,   # [mV]
                       
                       # CF determination
                       cubicSpline             = True,
                       cubicSplineRenderPoints = 200,
                       
                       # median filter
                       medianFilterA           = True,
                       windowSizeA             = 9,
                       medianFilterB           = True,
                       windowSizeB             = 5,
                       
                       debug                   = True)
        


