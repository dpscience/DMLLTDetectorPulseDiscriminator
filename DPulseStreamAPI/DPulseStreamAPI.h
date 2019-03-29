/****************************************************************************
**
**  Copyright (C) 2019 Danny Petschke
**
**  This program is free software: you can redistribute it and/or modify
**  it under the terms of the GNU General Public License as published by
**  the Free Software Foundation, either version 3 of the License, or
**  (at your option) any later version.
**
**  This program is distributed in the hope that it will be useful,
**  but WITHOUT ANY WARRANTY; without even the implied warranty of
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
**  GNU General Public License for more details.
**
**  You should have received a copy of the GNU General Public License
**  along with this program. If not, see http://www.gnu.org/licenses/.
**
*****************************************************************************
**
**  @author:  Danny Petschke
**  @contact: danny.petschke@uni-wuerzburg.de
**
*****************************************************************************/

#ifndef DPULSESTREAMMANAGER_H
#define DPULSESTREAMMANAGER_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string>
#include <fstream>

using namespace std;

#define DDELETE_SAFETY(__var__) if (__var__) { delete __var__; __var__ = nullptr; }

/****************************************************************************
**
**  struct DPulseStreamHeader:
**  
**  This struct is placed in the header (first lines) of each pulse stream file
**  using class DPulseStreamManager.
**  It contains the information, which are necessary to read this file afterwards. 
**
*****************************************************************************/

typedef struct {
	uint32_t     version;
	double       sweepInNanoseconds;
	double       sampleSpeedInGHz;
	int32_t      numberOfSamplePoints;
} DPulseStreamHeader;

#define sz_structDPulseStreamHeader sizeof(DPulseStreamHeader)

/****************************************************************************
**
**  class DPulseStreamManager:
**
**  This class provides the base functionality to stream the acquired detector 
**  output pulses to a binary file.
**
*****************************************************************************/

class DPulseStreamManager {
	DPulseStreamManager();
	virtual ~DPulseStreamManager();

	ofstream *m_file;
	string m_fileName;
	bool m_isArmed;

	__int64 m_contentInBytes;

public:
	static DPulseStreamManager *sharedInstance();

	/* this function creates the pulse stream binary file */
	bool start(const string& fileName, double sweepInNanoseconds, double sampleSpeedInGHz, int numberOfSamplePoints);

	/* this function closes the pulse stream binary file */
	void stopAndSave();

	/* this function streams ONE single pulse (time & voltage trace) to the binary file */
	/* int 'length': size of 'voltage' and/or 'time' in bytes */
	bool writePulse(float *time, float *voltage, int length);

	/* this function streams TWO pulses (time & voltage trace) to the binary file */
	/* int 'length': size of 'voltage' and/or 'time' in bytes */
	bool writePulsePair(float *time_1, float *voltage_1, float *time_2, float *voltage_2, int length);

	inline bool isArmed()    const { return m_isArmed; }
	inline string fileName() const { return m_fileName; }

	__int64 streamedContentInBytes() const { return m_contentInBytes; }
};

#endif // DPULSESTREAMMANAGER_H