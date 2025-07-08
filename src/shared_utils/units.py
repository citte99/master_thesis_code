import torch

c = 299792.458  # Speed of light in km/
G=4.3009172706e-6 #gravitational constant in kpc/M_sun*(km/s)^2


"""
Angular diameter distances: mega-parsecs
Angles: radians
Velocities: km/s
Mass: solar masses


"""


def _arcsec_to_rad(arcsec):
    return arcsec*torch.pi/648000.0

def _rad_to_arcsec(rad):
    return rad*648000.0/torch.pi