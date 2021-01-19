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

This example demonstrates how to run a ML pipeline on a 
TESTing and TRAINing set of streamed pulses (A/B).

A grid of prediction accuracies [0.0 - 1.0] on a given TEST set
is obtained by varying the number (N) of CORRECT TRAINed pulses vs. 
the number (N) of REJECT TRAINed pulses for a given machine's classifier:
    
[number of CORRECT TRAINed pulses x number of REJECT TRAINed pulses].

"""

########################## DEFINITIONS ##################################
#
relPath            = 'F:/'

"""
 define your TRAINing data set:
"""
filenameTrue_A     = relPath + 'A/true/A_2.drs4DataStream'
filenameFalse_A    = relPath + 'A/false/A_2.drs4DataStream'

filenameTrue_B     = relPath + 'B/true/B_2.drs4DataStream'
filenameFalse_B    = relPath + 'B/false/B_2.drs4DataStream'  

"""
 define your TESTing data set:
"""
_filenameTrue_A     = relPath + 'A/true/A.drs4DataStream'
_filenameFalse_A    = relPath + 'A/false/A.drs4DataStream'

_filenameTrue_B     = relPath + 'B/true/B.drs4DataStream'
_filenameFalse_B    = relPath + 'B/false/B.drs4DataStream'
#
#########################################################################

"""
 configure machine params of pulse (A):
"""
mlinputA = DMachineParams()

"""
 set classifier: 
 
 default: CalibratedClassifierCV(GaussianNB(), cv=2, method='isotonic')
"""
mlinputA.m_classifier = CalibratedClassifierCV(GaussianNB(), cv=2, method='isotonic')

# baseline correction
mlinputA.m_startCell  = 10
mlinputA.m_cellRegion = 150

# median filter
mlinputA.m_medianFilter = True
mlinputA.m_windowSize   = 5

"""
 configure machine params of pulse (B):
"""
mlinputB = DMachineParams()

"""
 set classifier: 
 
 default: CalibratedClassifierCV(GaussianNB(), cv=2, method='isotonic')
"""
mlinputB.m_classifier = CalibratedClassifierCV(GaussianNB(), cv=2, method='isotonic')

# baseline correction
mlinputB.m_startCell  = 10
mlinputB.m_cellRegion = 150

# median filter
mlinputB.m_medianFilter = True
mlinputB.m_windowSize   = 5

"""
 define the number of TESTing pulses used to calculate the prediction accuracy [0.0-1.0] of each 
 point (number of CORRECT pulses;number of REJECT pulses) on the mesh-grid. 
"""
numberOfPulsesN_test  = 1000

"""
 define the number of CORRECT/REJECT detector pulses for the TRAINING process:
     
 [2, 50, 2] means: looping from 2 to (=)50 with increment of 2.    
"""
pulseNRegion_corr = [2, 50, 2] 
pulseNRegion_rej  = [2, 50, 2]  

print('### pipeline grid for PULSE (A): {0}\n')
x_axis, y_axis, scoring, s = runPipelineGrid2(fileNameCorrectPulses_train = filenameTrue_A, 
                                              fileNameRejectPulses_train  = filenameFalse_A,
                                              fileNameCorrectPulses_test  = _filenameTrue_A, 
                                              fileNameRejectPulses_test   = _filenameFalse_A,
                                              numberOfPulsesCorrect_train = pulseNRegion_corr,
                                              numberOfPulsesReject_train  = pulseNRegion_rej,
                                              numberOfPulses_test         = numberOfTESTPulses,
                                              isPositivePolarity          = False,
                                              machineInput                = mlinputA)

"""
 visualize the prediction accuracy mesh-grid of PULSE (A):
"""
print('best prediction accuracy (A):')
print(s)
pl.title('pipeline grid of PULSE (A): number of CORRECT pulses vs. number of REJECT pulses')
pl.xlabel('number of CORRECT TRAINed pulses [#]')
pl.ylabel('number of REJECT TRAINed pulses [#]')

scale = pl.contourf(x_axis, y_axis, scoring, levels=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0], cmap="jet") 
pl.colorbar(scale)
pl.show()

print('### pipeline grid for PULSE (B): {0}\n')
x_axis, y_axis, scoring, s = runPipelineGrid2(fileNameCorrectPulses_train = filenameTrue_B, 
                                              fileNameRejectPulses_train  = filenameFalse_B,
                                              fileNameCorrectPulses_test  = _filenameTrue_B, 
                                              fileNameRejectPulses_test   = _filenameFalse_B,
                                              numberOfPulsesCorrect_train = pulseNRegion_corr,
                                              numberOfPulsesReject_train  = pulseNRegion_rej,
                                              numberOfPulses_test         = numberOfTESTPulses,
                                              isPositivePolarity          = False,
                                              machineInput                = mlinputB)

"""
 visualize the accuracy mesh-grid of PULSE (B):
"""
print('best prediction accuracy (B):')
print(s)
pl.title('pipeline grid of PULSE (B): number of CORRECT pulses vs. number of REJECT pulses')
pl.xlabel('number of TRAINed CORRECT pulses [#]')
pl.ylabel('number of TRAINed REJECT pulses [#]')

scale = pl.contourf(x_axis, y_axis, scoring, levels=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0], cmap="jet") 
pl.colorbar(scale)
pl.show()   

