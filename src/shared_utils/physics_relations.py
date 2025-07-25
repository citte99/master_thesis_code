import torch
from astropy import units as u
import numpy as np
from astropy.constants import G



# NOTE: PREVIOUS MESSY STUPID, learn math please. BUT STILL WAS CORRECT
def r_max_moline(M_max : torch.Tensor)-> torch.Tensor : #not very precise type suggestion
    from astropy.constants import G
    G=G.to(u.kpc**3/u.M_sun*u.s**(-2))
    
    A=0.344*u.kpc #kpc
    B=1.607

    M_max = M_max * u.M_sun  # Convert to astropy units
    
    const_1=(np.log(2.163+1.)+1./(2.163+1)-1)*4.*np.pi
    
    #print(const_1)
        
    const_2=(    10* u.km/u.s     /1.64/np.sqrt(G)   *(2.163/A)**(1/B)    )**2
    #print(const_2)
    
    
    r_s=((M_max/const_1/const_2).to(u.kpc**(1 + 2/B)))**(B/(B+2))
    
    r_max=2.163*r_s
    
    return r_max.to(u.kpc).value



# G=G.to(u.kpc**3/u.M_sun*u.s**(-2))

# A = 0.344 * u.kpc
# B = 1.607

# def r_max_moline(M_max):
#     """
#     Compute r_max (in kpc) given M_max (in solar masses)
#     using the Moliné et al. v_max-r_max relation + Newton's law.
#     """
#     M = M_max * u.M_sun
#     # exponent in the analytic solution:
#     α = B / (B + 2)

#     # invert M = v^2 r / G with v = 10 * (r/A)^ (1/B)
#     # => r_max = [ G M A^(2/B) / (10 km/s)^2 ]^(B/(B+2))
#     r = (G * M * A**(2/B) / ( (10*u.km/u.s)**2 ))**α

#     return r.to(u.kpc).value



def test():
    import numpy as np
    test_cases = [
        (1e6,      0.16), # log mass, and kpc
        (10**7.4,  0.56),
        (10**8.6,  1.92),
        (10**9.8,  6.56),
        (10**11.0, 22.27),
    ]
    M_expec     = np.array([test_cases[i][0] for i in range(len(test_cases))])
    r_max_expec = np.array([test_cases[i][1] for i in range(len(test_cases))])


    r = r_max_moline(M_expec)
    print(r),
    print(r_max_expec)
    assert np.isclose(r, r_max_expec, rtol = 1e-2, atol = 1e-2).all(), f'test failed in {__file__}. I blame this on conor\'s paper'


