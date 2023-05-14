import numpy as np
from numpy import pi, cos, sin, sqrt, log
from scipy import constants as const
import math
import scipy
import glob

'''
Some code for frequency estimation
'''

class unique_element:
    def __init__(self,value,occurrences):
        self.value = value
        self.occurrences = occurrences
        
def perm_unique(elements):
    eset=set(elements)
    listunique = [unique_element(i,elements.count(i)) for i in eset]
    u=len(elements)
    return perm_unique_helper(listunique,[0]*u,u-1)

def perm_unique_helper(listunique,result_list,d):
    if d < 0:
        yield tuple(result_list)
    else:
        for i in listunique:
            if i.occurrences > 0:
                result_list[d]=i.value
                i.occurrences-=1
                for g in  perm_unique_helper(listunique,result_list,d-1):
                    yield g
                i.occurrences+=1

def TE_mode_sort(mode_num=1):
    '''
    This calculates allowable nml values for the TE modes of the resonator. Uses above functions
    '''
    
    permutes=[[1,0,1],[1,0,2],[1,0,3],[1,0,4],[1,0,5],[2,0,2],[2,0,3],[2,0,4],[2,0,5],[3,0,3],[3,0,4],[3,0,5]]
    nml=[]
    for i in range(len(permutes)):
        perms=np.array(list(perm_unique(permutes[i])))
        for vals in perms:
            if vals[0]!=0 and vals[1]==0 and vals[2]!=0:
                nml.append(list(vals))
            else:
                pass
    nml=np.array(nml)
    i_sort=np.argsort(nml[:,0])
    return nml[i_sort][0:mode_num]


def freq_rect(a,b,c, modes=1, unit='metric'):
    '''
    Calculate the lowest n-mode frequencies for a rectangular cavity of a,b,c dimensions where b is the smallest dim
    '''
    if unit=='metric':
        a=a/1000.
        b=b/1000.
        c=c/1000.
    elif unit=='imperial':
        a=a*.0254
        b=b*.0254
        c=c*.0254
    else:
        raise Exception('ERROR: Unit must be metric or imperial')
    
    eps=const.epsilon_0
    mu=const.mu_0
    c_speed=const.c
    nml_array=TE_mode_sort(10)
    f=[]

    for vals in nml_array:
        n=vals[0]
        m=vals[1]
        l=vals[2]
        f.append(((c_speed)/(2*pi))*sqrt((n*pi/a)**2+(m*pi/b)**2+(l*pi/c)**2))
    f=np.array(f)
    f_sort=np.argsort(f)
    return f[f_sort][0:modes], nml_array[f_sort]

def get_c_dim(f,a,b, unit='metric'):
    '''
    Estimates the second long dimension based on a target frequency and the other two dimensions. Works only for the 
    fundamental mode
    '''
    brentq=scipy.optimize.brentq
    f_intercept=lambda c:freq_rect(a,b,c,1, unit)[0][0]-f
    c=brentq(f_intercept, min([10*a, b]), max([10*a, b]))

    return c

def round_dim(dim, frac):
    val=frac*math.ceil(dim/frac)+frac
    if (val-dim)/2<frac:
        val=val+.25
    else:
        pass
    return val

def val_gen(val_in, units='metric', scale='mm'):
    if units=='imperial':
        fac=1
        unit='in'
    elif units=='metric':
        if scale=='mm':
            fac=25.4
        elif scale=='m':
            fac=.0254
    else:
        raise Exception('Wrong unit val')
        
    if type(val_in)==list:
        for I, val in enumerate(val_in):
            val_in[I]=fac*val
    else:
        val_in=fac*val_in
    
    return val_in

def check_path(path):
    check=glob.glob(path)
    if check==[]:
        return False
    else:
        return True

def cap_sigma(cmat, units='fF'):
    cmat=Q(cmat, units).to('F').magnitude
    return (abs(cmat[0,1])+(cmat[0,0]*cmat[1,1])/(cmat[0,0]+cmat[1,1]))

def C_to_Ec(cmat, cap_units='fF'):
    csig=cap_sigma(cmat, cap_units)
    return const.e**2/(2*(csig))/const.h