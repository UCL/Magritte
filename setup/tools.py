import numpy as np


# Physical constants
c     = 2.99792458E+08   # [m/s] speed of light
h     = 6.62607004E-34   # [J*s] Planck's constant
kb    = 1.38064852E-23   # [J/K] Boltzmann's constant
amu   = 1.66053904E-27   # [kg] atomic mass unit
T_CMB = 2.72548000E+00   # [K] CMB temperature


def LTEpop (linedata, temperature):
    '''
    Return the LTE level populations give the temperature
    '''
    pop = np.zeros(linedata.nlev)
    # Calculate the LTE populations
    for i in range(linedata.nlev):
        pop[i] = linedata.weight[i] * np.exp(-linedata.energy[i]/(kb*temperature))
    # Normalize to relative populations
    pop = pop / sum(pop)
    # Done
    return pop


def lineEmissivity (linedata, pop):
    '''
    Return the line emisivvity for each radiative transition
    '''
    eta = np.zeros(linedata.nrad)
    for k in range(linedata.nrad):
        i = linedata.irad[k]
        j = linedata.jrad[k]
        eta[k] = h*linedata.frequency[k]/(4.0*np.pi) * linedata.A[k]*pop[i]
    # Done
    return eta


def lineOpacity (linedata, pop):
    '''
    Return the line opacity for each radiative transition
    '''
    chi = np.zeros(linedata.nrad)
    for k in range(linedata.nrad):
        i = linedata.irad[k]
        j = linedata.jrad[k]
        chi[k] = h*linedata.frequency[k]/(4.0*np.pi) * (linedata.Ba[k]*pop[j] - linedata.Bs[k]*pop[i])
    # Done
    return chi


def lineSource (linedata, pop):
    S = lineEmissivity (linedata, pop) / lineOpacity (linedata, pop)
    # Done
    return S


def planck (temperature, frequency):
    '''
    Planck function for thermal radiation.
    '''
    return 2.0*h/c**2 * np.power(frequency,3) / np.expm1(h*frequency/(kb*temperature))



def dnu (nu_ij, inverse_mass, temp, turb):
    """
    :param nu_ij: line centre frequency
    :param inverse_mass: inverse mass of the line producing species
    :param temp: temperature
    :param turb: turbulent velocity
    :return: line width of the line profile function.
    """
    return nu_ij/c * np.sqrt(2.0*kb*temp/amu*inverse_mass + turb**2)

def profile (nu_ij, inverse_mass, temp, turb, nu):
    """
    :param nu_ij: line centre frequency
    :param inverse_mass: inverse mass of the line producing species
    :param temp: temperature
    :param turb: turbulent velocity
    :param nu: frequency at which to evaluate
    :return: Gaussian profile function evaluated at frequency nu.
    """
    return np.exp(-((nu-nu_ij)/dnu(nu_ij, inverse_mass, temp, turb))**2) / (np.sqrt(np.pi) * dnu(nu_ij, inverse_mass, temp, turb))
