![badge-OS](https://img.shields.io/badge/OS-Windows-blue)
![badge-language](https://img.shields.io/badge/language-Python-blue)
![badge-license](https://img.shields.io/badge/license-BSD-blue)

Support this project and keep always updated about recent software releases, bug fixes and major improvements by [following on researchgate](https://www.researchgate.net/project/A-supervised-machine-learning-approach-for-shape-sensitive-detector-pulse-discrimination-in-lifetime-spectroscopy-applications) or [github](https://github.com/dpscience?tab=followers).

[![badge-researchGate](https://img.shields.io/badge/project-researchGate-brightgreen)](https://www.researchgate.net/project/A-supervised-machine-learning-approach-for-shape-sensitive-detector-pulse-discrimination-in-lifetime-spectroscopy-applications)

![badge-followers](https://img.shields.io/github/followers/dpscience?style=social)
![badge-stars](https://img.shields.io/github/stars/dpscience/DMLLTDetectorPulseDiscriminator?style=social)
![badge-forks](https://img.shields.io/github/forks/dpscience/DMLLTDetectorPulseDiscriminator?style=social)


# DMLLTDetectorPulseDiscriminator Framework 

Copyright (c) 2019-2021 Danny Petschke (danny.petschke@uni-wuerzburg.de). All rights reserved.<br><br>
<b>DMLLTDetectorPulseDiscriminator</b> - A supervised machine learning approach for shape-sensitive detector pulse discrimination in lifetime spectroscopy applications.
<br>

![DMLLTDetectorPulseDiscriminatorAbstract](/NIMpub.png)

# Content of the Framework
 
## ``pyDMLLTDetectorPulseDiscriminator``

A <b>python-based framework</b>providing the functionalities for <b>TRAINing/EVALuating and TESTing a classifier</b> such as the naive Bayes classifier on a set of CORRECT and FALSE/REJECT detector output pulses. Moreover, it provides the generation of lifetime spectra from shape-discriminated detector pulses applying the TRAINed classifier, i.e. the classifier providing the highest prediction accuracy.

## ``DPulseStreamAPI``

A simple exchange protocol written in native C++ providing the functionality for streaming the acquired detector output pulses on a mass storage device according to the data format required from the [pyDMLLTDetectorPulseDiscriminator](https://github.com/dpscience/DMLLTDetectorPulseDiscriminator/pyDMLLTDetectorPulseDiscriminator) framework for TRAINing/EVALuating and TESTing the classifiers. This protocol is not attached to any type of digitizer enabling an universal application of this approach. 

# Quickstart Guide

## ``Basic Principle``

* <b>Store a number of CORRECT (OK) and FALSE (!OK) assigned output pulses</b> each in a separate stream for both detectors A and B. In case of you are using the [DRS4 evaluation board](https://www.psi.ch/en/drs/evaluation-board) of the Paul-Scherrer Institute (PSI, Switzerland) the software tool [DDRS4PALS](https://github.com/dpscience/DDRS4PALS) does this job for you.<br><br>
* <b>TRAIN/EVALuate and TEST the classifiers</b> for A and B on this set of streamed detector pulses.<br><br>
For more details see [examples](https://github.com/dpscience/DMLLTDetectorPulseDiscriminator/tree/master/pyDMLLTDetectorPulseDiscriminator/examples).
<br><br>
* <b>Finally, generate the lifetime spectrum</b> using the pulse stream recorded on the studied sample material by applying the TRAINed classifiers for pulse-shape discrimination.<br><br>  

![DMLLTDetectorPulseDiscriminator](/principle.png)

## ``How to apply? ... only a few Lines of Code ...`` 

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
 pulse stream acquired of the studied sample which is used to generate the lifetime spectrum:
"""
dataStream         = 'E:/Fe/pure_iron.drs4DataStream'

"""
 location and filename where the resulting lifetime spectrum is stored:
"""
spectraOutput      = relPath + 'spectrum/spectrum_Fe'

"""
 configure the machine params of pulse (A):
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
 configure the machine params of pulse (B):
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
 define the number of labelled detector pulses (A/B) which should be used to TRAIN the machine's classifier prior to the lifetime spectrum generation:
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
# Related Publication/Presentation

### ``Publication in NIM A (Dec. 2019)``
[A supervised machine learning approach using naive Gaussian Bayes classification for shape-sensitive detector pulse discrimination in positron annihilation lifetime spectroscopy (PALS) (Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment, Elsevier)](https://www.sciencedirect.com/science/article/pii/S0168900219312021?via%3Dihub)<br><br>

### ``Presentation at 15th International Workshop on Slow Positron Beam Techniques & Applications (SLOPOS-15) in Prague (Sept. 2019)``

[SLOPOS-15 (Prague): A supervised machine learning approach for shape-sensitive detector pulse discrimination in positron spectroscopy applications](https://www.researchgate.net/publication/339573579_SLOPOS-15_Prague_A_supervised_machine_learning_approach_for_shape-sensitive_detector_pulse_discrimination_in_positron_spectroscopy_applications)<br><br>

# How to cite this Framework?

* <b>You should at least cite the following publication.</b><br>

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.nima.2019.162742-yellowgreen)](https://doi.org/10.1016/j.nima.2019.162742)

[A supervised machine learning approach using naive Gaussian Bayes classification for shape-sensitive detector pulse discrimination in positron annihilation lifetime spectroscopy (PALS) (Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment, Elsevier)](https://www.sciencedirect.com/science/article/pii/S0168900219312021?via%3Dihub)<br>

* <b>Additionally, you must cite the applied version of the framework in your study.</b><br>

You can cite all released software versions by using the <b>DOI 10.5281/zenodo.2616929</b>. This DOI represents all versions, and will always resolve to the latest one.<br>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2616929.svg)](https://doi.org/10.5281/zenodo.2616929)

## ``v1.x``

<b>DMLLTDetectorPulseDiscriminator v1.0</b><br>[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2616929.svg)](https://doi.org/10.5281/zenodo.2616929)<br>

# License of pyDMLLTDetectorPulseDiscriminator (BSD-3-Clause)

Copyright (c) 2019-2021 Danny Petschke (danny.petschke@uni-wuerzburg.de). All rights reserved.<br>

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
 
 # License of DPulseStreamAPI (GNU General Public License)
 
 Copyright (c) 2019-2021 Danny Petschke (danny.petschke@uni-wuerzburg.de) All rights reserved.<br><br>

<p align="justify">This program is free software: you can redistribute it and/or modify<br>
it under the terms of the GNU General Public License as published by<br>
the Free Software Foundation, either version 3 of the License, or<br>
(at your option) any later version.<br><br>

This program is distributed in the hope that it will be useful,<br>
but WITHOUT ANY WARRANTY; without even the implied warranty of<br>
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.<br><br></p>

For more details see [GNU General Public License v3](https://www.gnu.org/licenses/gpl-3.0)


