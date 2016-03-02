# Plot GaussPy results
import numpy as np
import matplotlib.pyplot as plt
import pickle
    
def gaussian(amp, fwhm, mean):
    sigma = np.float(fwhm / (2. * np.sqrt(2. * np.log(2))))
    return lambda x: amp * np.exp(-(x-mean)**2/2./sigma**2)
    
def unravel(list):
    return np.array([i for array in list for i in array])
    
DATA = 'simple_gaussian.pickle'
DATA_decomposed = 'simple_gaussian_decomposed.pickle'
    
data = pickle.load(open(DATA))
spectrum = np.array(unravel(data['data_list']))
chan = np.array(unravel(data['x_values']))
errors = np.array(unravel(data['errors']))
    
decomposed_data = pickle.load(open(DATA_decomposed))
means_fit = unravel(decomposed_data['means_fit'])
amps_fit = unravel(decomposed_data['amplitudes_fit'])
fwhms_fit = unravel(decomposed_data['fwhms_fit'])

fig = plt.figure()
ax = fig.add_subplot(111)
    
model = np.zeros(len(chan))

for j in range(len(means_fit)):
    component = gaussian(amps_fit[j], fwhms_fit[j], means_fit[j])(chan)
    model += component
    ax.plot(chan, component, color='red', lw=1.5)

ax.plot(chan, spectrum, label='Data', color='black', linewidth=1.5)
ax.plot(chan, model, label = r'$\alpha=10.$', color='purple', linewidth=2.)
ax.plot(chan, errors, label = 'Errors', color='green', linestyle='dashed', linewidth=2.)

ax.set_xlabel('Channels')
ax.set_ylabel('Amplitude')

ax.set_xlim(0,len(chan))
ax.set_ylim(np.min(spectrum),np.max(spectrum))
ax.legend(loc=2)

plt.show()