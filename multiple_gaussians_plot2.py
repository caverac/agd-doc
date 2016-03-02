# Decompose multiple Gaussian dataset using AGD
import pickle
import gausspy.gp as gp

# Specify necessary parameters
alpha1 = 20.
snr_thresh = 5.
DATA = 'multiple_gaussians.pickle'
DATA_out = 'multiple_gaussians_decomposed.pickle'

# Load GaussPy
g = gp.GaussianDecomposer()
    
# Setting AGD parameters
g.set('phase', 'one')
g.set('SNR_thresh', [snr_thresh, snr_thresh])
g.set('alpha1', alpha1)
g.set('mode','conv')
    
# Run GaussPy
decomposed_data = g.batch_decomposition(DATA)

# Save decomposition information
pickle.dump(decomposed_data, open(DATA_out, 'w'))

#quit()

import numpy as np
import matplotlib.pyplot as plt

def gaussian(amp, fwhm, mean):
    sigma = np.float(fwhm / (2. * np.sqrt(2. * np.log(2))))
    return lambda x: amp * np.exp(-(x-mean)**2/2./sigma**2)

def unravel(list):
    return np.array([i for array in list for i in array])

alpha1 = 20.
alpha2 = 2.
alpha3 = 10.

#alpha=10
decomposed_data1 = pickle.load(open('multiple_gaussians_decomposed1.pickle'))
means_fit1 = unravel(decomposed_data1['means_fit'])
amps_fit1 = unravel(decomposed_data1['amplitudes_fit'])
fwhms_fit1= unravel(decomposed_data1['fwhms_fit'])

#--alpha=6
decomposed_data2 = pickle.load(open('multiple_gaussians_decomposed2.pickle'))
means_fit2 = unravel(decomposed_data2['means_fit'])
amps_fit2 = unravel(decomposed_data2['amplitudes_fit'])
fwhms_fit2= unravel(decomposed_data2['fwhms_fit'])

#--alpha=8
decomposed_data3 = pickle.load(open('multiple_gaussians_decomposed3.pickle'))
means_fit3 = unravel(decomposed_data3['means_fit'])
amps_fit3 = unravel(decomposed_data3['amplitudes_fit'])
fwhms_fit3= unravel(decomposed_data3['fwhms_fit'])

data = pickle.load(open(DATA))
spectrum = np.array(unravel(data['data_list']))
chan = np.array(unravel(data['x_values']))
errors = np.array(unravel(data['errors']))

# plot results for the three alpha valubes
fig, (ax1, ax2,ax3) = plt.subplots(1, 3,figsize=(8,4), sharey=True)

#plt.figure(1,[8,3])
#ax1 = plt.subplot(1,3,1)
#ax2 = plt.subplot(1,3,2)
#ax3 = plt.subplot(1,3,3)

model1 = chan * 0.
for j in range(len(means_fit1)):
    component = gaussian(amps_fit1[j], fwhms_fit1[j], means_fit1[j])(chan)
    model1 += component
    ax1.plot(chan, component, color='red', lw=1.5)

ax1.plot(chan, spectrum, label='Spectrum', color='black', lw=1.5)
ax1.plot(chan, model1, label = r'GaussPy: $\alpha=$'+str(alpha1), color='purple', lw=2.)
ax1.plot(chan, errors, label = 'Errors', color='green', ls='dashed', lw=2.)

ax1.legend(loc=1, fontsize=8)
ax1.set_xlabel('Channels')
ax1.set_ylabel('Amplitude')
ax1.set_xlim(0,len(chan))
ax1.set_ylim(np.min(spectrum),np.max(spectrum))

model2 = chan * 0.
for j in range(len(means_fit2)):
    component2 = gaussian(amps_fit2[j], fwhms_fit2[j], means_fit2[j])(chan)
    model2 += component2
    ax2.plot(chan, component2, color='red', lw=1.5)

ax2.plot(chan, spectrum, label='Spectrum', color='black', lw=1.5)
ax2.plot(chan, model2, label = r'GaussPy: $\alpha=$'+str(alpha2), color='purple', lw=2.)
ax2.plot(chan, errors, label = 'Errors', color='green', ls='dashed', lw=2.)
ax2.legend(loc=1, fontsize=8)
ax2.set_xlabel('Channels')
ax2.set_xlim(0,len(chan))
ax2.set_ylim(np.min(spectrum),np.max(spectrum))

model3 = chan * 0.
for j in range(len(means_fit3)):
    component3 = gaussian(amps_fit3[j], fwhms_fit3[j], means_fit3[j])(chan)
    model3 += component3
    ax3.plot(chan, component3, color='red', lw=1.5)

ax3.plot(chan, spectrum, label='Spectrum', color='black', lw=1.5)
ax3.plot(chan, model3, label = r'GaussPy: $\alpha=$'+str(alpha3), color='purple', lw=2.)
ax3.plot(chan, errors, label = 'Errors', color='green', ls='dashed', lw=2.)
ax3.legend(loc=1, fontsize=8)
ax3.set_xlabel('Channels')
ax3.set_xlim(0,len(chan))
ax3.set_ylim(np.min(spectrum),np.max(spectrum))

plt.tight_layout()
plt.show()