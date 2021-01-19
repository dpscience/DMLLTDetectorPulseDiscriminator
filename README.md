Support this project and keep always updated about recent software releases, bug fixes and major improvements by [following on researchgate](https://www.researchgate.net/project/A-supervised-machine-learning-approach-for-shape-sensitive-detector-pulse-discrimination-in-lifetime-spectroscopy-applications) or [github](https://github.com/dpscience?tab=followers).<br><br>

# DMLLTDetectorPulseDiscriminator  

Copyright (c) 2019 Danny Petschke (danny.petschke@uni-wuerzburg.de). All rights reserved.<br><br>
<b>DMLLTDetectorPulseDiscriminator</b> - A supervised Machine Learning Approach for shape-sensitive Detector Pulse Discrimination in Lifetime Spectroscopy Applications.
<br>

# Software Packages and Requirements

## (1) pyDMLLTDetectorPulseDiscriminator:

A <b>python-based framework</b>, which provides the functionalities for <b>TRAINing and TESTing/EVALuating a classifier</b>, such as the naive Bayes classifier, on a set of CORRECT and FALSE/REJECT detector output pulses. Moreover, it provides the generation of lifetime spectra from shape-discriminated detector pulses applying the TRAINed classifier.

## Requirements

- [scikit-learn](https://scikit-learn.org/stable/)
- [NumPy](http://www.numpy.org/) 
- [SciPy](https://www.scipy.org/)
- [matplotlib](https://matplotlib.org/)<br>
- [joblib](https://joblib.readthedocs.io/en/latest/)

#### [WinPython](https://sourceforge.net/projects/winpython/) meets all requirements. 

## (2) DPulseStreamAPI: 

A simple exchange protocol written in C++ providing the functionality for streaming the acquired detector output pulses on a mass storage device according to the format required by the [pyDMLLTDetectorPulseDiscriminator](https://github.com/dpscience/DMLLTDetectorPulseDiscriminator/pyDMLLTDetectorPulseDiscriminator) framework for training and testing the calssifiers. 

# How to start?

## Basic Principle

<b>(1) Store a number of CORRECT and FALSE assigned output pulses</b> on a separate stream for each detector A and B (e.g. in case of the DRS4 evaluation board you can easily use [DDRS4PALS software](https://github.com/dpscience/DDRS4PALS).<br><br>
<b>(2) TRAIN (and TEST/EVALuate) the machine's classifier</b> (A and B) on a set of (CORRECT/FALSE) streamed detector pulses.<br><br>
For more details see [examples](https://github.com/dpscience/DMLLTDetectorPulseDiscriminator/tree/master/pyDMLLTDetectorPulseDiscriminator/examples).
<br><br><b>(3) Generate the lifetime spectrum</b> using the pulse stream recorded on the studied sample material by applying the TRAINed classifiers for pulse-shape discrimination.<br><br>  

![DMLLTDetectorPulseDiscriminator](/principle.png)

```python
relPath            = 'C:/dpscience/'

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
dataStream         = 'E:/Fe/pure_iron.drs4DataStream'

"""
 location and filename where the resulting lifetime spectrum will be stored:
"""
spectraOutput      = relPath + 'spectrum/spectrum_Fe'

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
```
# Related Publications

### Dec. 2019
[A supervised machine learning approach using naive Gaussian Bayes classification for shape-sensitive detector pulse discrimination in positron annihilation lifetime spectroscopy (PALS) (Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment, Elsevier)](https://www.sciencedirect.com/science/article/pii/S0168900219312021?via%3Dihub)<br><br>

# How to cite this Software?

<b>You should at least cite the following publication:</b><br><br>
[A supervised machine learning approach using naive Gaussian Bayes classification for shape-sensitive detector pulse discrimination in positron annihilation lifetime spectroscopy (PALS) (Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment, Elsevier)](https://www.sciencedirect.com/science/article/pii/S0168900219312021?via%3Dihub)<br>

Additionally, you can cite all released software versions by using the the <b>DOI 10.5281/zenodo.2616929</b>. This DOI represents all versions, and will always resolve to the latest one.<br>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2616929.svg)](https://doi.org/10.5281/zenodo.2616929)

## v1.x

<b>DMLLTDetectorPulseDiscriminator v1.0</b><br>[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2616929.svg)](https://doi.org/10.5281/zenodo.2616929)<br>

# (BSD-3-Clause) - pyDMLLTDetectorPulseDiscriminator

Copyright (c) 2019 Danny Petschke (danny.petschke@uni-wuerzburg.de). All rights reserved.<br>

Redistribution and use in source and binary forms, with or without modification,<br> 
are permitted provided that the following conditions are met:<br><br>

 1. Redistributions of source code must retain the above copyright notice<br>
    this list of conditions and the following disclaimer.<br><br>

 2. Redistributions in binary form must reproduce the above copyright notice,<br> 
    this list of conditions and the following disclaimer in the documentation<br> 
    and/or other materials provided with the distribution.<br><br>

 3. Neither the name of the copyright holder "Danny Petschke" nor the names of<br> 
    its contributors may be used to endorse or promote products derived from <br>
    this software without specific prior written permission.<br><br>


 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS<br> 
 OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF<br> 
 MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE<br> 
 COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,<br> 
 EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF<br> 
 SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)<br> 
 HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR<br> 
 TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,<br> 
 EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.<br>
 
 For more details see [BSD-3-Clause License](https://opensource.org/licenses/BSD-3-Clause)
 
 # (GNU General Public License) - DPulseStreamAPI
 
 Copyright (c) 2019 Danny Petschke (danny.petschke@uni-wuerzburg.de) All rights reserved.<br><br>

<p align="justify">This program is free software: you can redistribute it and/or modify<br>
it under the terms of the GNU General Public License as published by<br>
the Free Software Foundation, either version 3 of the License, or<br>
(at your option) any later version.<br><br>

This program is distributed in the hope that it will be useful,<br>
but WITHOUT ANY WARRANTY; without even the implied warranty of<br>
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.<br><br></p>

For more details see [GNU General Public License v3](https://www.gnu.org/licenses/gpl-3.0)


