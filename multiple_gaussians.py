# Create profile with multiple, blended Gaussians and added noise
# Store in format required for GaussPy

import numpy as np
import pickle

def gaussian(amp, fwhm, mean):
    return lambda x: amp * np.exp(-(x-mean)**2/4./(fwhm/2.355)**2)

# Specify filename of output data
FILENAME = 'multiple_gaussians.pickle'

# Number of Gaussian functions per spectrum
NCOMPS = 3

# Component properties
AMPS = [3,2,1]
FWHMS = [10,50,30] # channels
MEANS = [210,250,310] # channels

# Data properties
RMS = 0.05
NCHANNELS = 512

# Initialize
gausspy_data = {}
chan = np.arange(NCHANNELS)
errors = np.ones(NCHANNELS) * RMS

spectrum = np.random.randn(NCHANNELS) * RMS

# Create spectrum
for a, w, m in zip(AMPS, FWHMS, MEANS):
    spectrum += gaussian(a, w, m)(chan)

# Enter results into AGD dataset
gausspy_data['data_list'] = gausspy_data.get('data_list', []) + [spectrum]
gausspy_data['x_values'] = gausspy_data.get('x_values', []) + [chan]
gausspy_data['errors'] = gausspy_data.get('errors', []) + [errors]

pickle.dump(gausspy_data, open(FILENAME, 'w'))
#print 'Created: ', FILENAME

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(chan, spectrum, color='black', label='Spectrum', linewidth=1.5)
ax.plot(chan, errors, color='green', label='Errors', linestyle='dashed', linewidth=2.)
ax.axhline(0.0, color='black', lw=1)

ax.set_title(r'Multiple Gaussians Example')
ax.set_xlabel(r'Channels')
ax.set_ylabel(r'Amplitude')

ax.set_xlim(0,NCHANNELS)
ax.set_ylim(np.min(spectrum),np.max(spectrum))
plt.legend(loc=2)

plt.show()