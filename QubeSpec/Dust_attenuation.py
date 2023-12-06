"""
Example: Ext
from ext import extinct_car, wavelengths

wa=wavelengths()
OIIIWav, HaWav, HbWav, NIIWav, OIIWav, SIIWav=wa.wavs()
OIII, Av1=extinct_car(self.OIII, Ha1, Hb1, OIIIWav, 3.1)
NII, Av2=extinct_car(self.NII, Ha1, Hb1, NIIWav, 3.1)
"""

__author__='Jan Scholtz, William M. Baker, Ignas'

import numpy as np
from numba import jit, float32


Balmer_lines = {'Halpha': [6562.819e-10, 2.863]}
Balmer_lines['Hbeta'] = [4861.333e-10, 1]
Balmer_lines['Hgamma'] = [4340e-10, 0.468]
Balmer_lines['Hdelta'] = [4101.734e-10, 0.259]
Balmer_lines['H10'] = [3970.075e-10, 0.0533]



class Model:
    def __init__(self,kb1s, kb2s, Balmer_rats, ):
        self.kb1= kb1s
        self.kb2 = kb2s
        self.Balmer_rats = Balmer_rats

        
    def rat_calc(self, x, Av):
        dkb = self.kb1-self.kb2
        exp = dkb*Av/-2.5
        #Av1= -2.5 * np.log10((fb1/fb2)/Balmer_rat) * 1/(Kb1- Kb2)
        rats = 10**(exp)*self.Balmer_rats
        return rats

class Dust_cor:

    def __init__(self): 
        self.Balmer_lines = Balmer_lines

    
    def flux_cor(self, F, wav, fb1, fb2, fb_names, R_v=None, curve='smc', curve_fce=None):
        fb1_name, fb2_name = fb_names.split('_')
        self.R_v = R_v
        
        fb1_wave = self.Balmer_lines[fb1_name][0]
        fb2_wave = self.Balmer_lines[fb2_name][0]

        Balmer_rat = self.Balmer_lines[fb1_name][1]/self.Balmer_lines[fb2_name][1]
        if curve_fce !=None:
            if curve=='smc':
                curve_fce = self.smc
            elif curve=='calzetti2000':
                curve_fce = self.calzetti2000
            elif curve=='cardonelli1989':
                curve_fce = self.cardelli1989

        Kb1=curve_fce(fb1_wave, R_v = R_v)
        Kb2=curve_fce(fb2_wave, R_v = R_v)
        K=curve_fce(wav*1e-10, R_v = R_v)
        Av=[]
        f=[]

        for i in range(F.shape[0]):
            try:
                Av1= -2.5 * np.log10((fb1[i]/fb2[i])/Balmer_rat) * K/(Kb1- Kb2)
                f1=F[i]*10**(Av1/2.5)
                Av.append(Av1)
                f.append(f1)
            except Exception as _exc_:
                print(_exc_)
                f.append(np.nan)
                Av.append(np.nan)
        f=np.array(f)
        Av=np.array(Av)
        return f, Av
    
    def flux_cor_fit(self, F, wav, ratios, fb_names,eratios=None, R_v=None, curve='smc', curve_fce=None):

        if curve_fce !=None:
            if curve=='smc':
                curve_fce = self.smc
            elif curve=='calzetti2000':
                curve_fce = self.calzetti2000
            elif curve=='cardonelli1989':
                curve_fce = self.cardelli1989
        self.R_v = R_v

        Balmer_rats = []
        Kb1s = []
        Kb2s = []
        for name in fb_names:
            fb1_name, fb2_name = name.split('_')
            
            Balmer_rats.append(self.Balmer_lines[fb1_name][1]/self.Balmer_lines[fb2_name][1])
            
            fb1_wave = self.Balmer_lines[fb1_name][0]
            fb2_wave = self.Balmer_lines[fb2_name][0]

        
            Kb1s.append(self.curve_fce(fb1_wave, R_v = self.R_v))
            Kb2s.append(self.curve_fce(fb2_wave, R_v = self.R_v))
        Kb1s= np.array(Kb1s)
        Kb2s = np.array(Kb2s)
        K = self.curve_fce(wav*1e-10, R_v = self.R_v)

        from scipy.optimize import curve_fit
        Ext_model = Model(Kb1s, Kb2s, Balmer_rats)
        Av_fit, err  = curve_fit(Ext_model.rat_calc, np.zeros_like(ratios), ratios, p0=0.5, sigma=eratios)
        
        Av = Av_fit*K
        f=F*10**(Av/2.5)
        return f, Av, Av_fit, np.sqrt(err)
    
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def calzetti2000(wav, R_v=None):
        if R_v:
            R_v = R_v
        else: 
            R_v = 3.1
        wav=wav/(1.0e-6)
        if 0.12 < wav < 0.63:
            return 2.659 * (-2.156 + 1.509/wav - 0.198/(wav**2) 
                                                +0.011/(wav **3)) + R_v
        elif 0.63 < wav < 2.20:
            return 2.659*(-1.857 + 1.040/wav ) + R_v
        else:
            return 0

    @staticmethod
    @jit(nopython=True, cache=True)
    def F_a(x):
        if x < 5.9:
            return 0.
        elif 5.9 < x <8.:
            return -0.04473 * ( x - 5.9)**2 - 0.009779 * ( x- 5.9 )**3

    @staticmethod
    @jit(nopython=True, cache=True)
    def F_b(x):
        if x < 5.9:
            return 0.
        elif 5.9 < x <8.:
            return 0.2130 * ( x - 5.9)**2 + 0.1207 * ( x- 5.9 )**3

    @staticmethod
    @jit(nopython=True, cache=True)
    def cardelli1989(wav, R_v=None):
        if R_v:
            R_v = R_v
        else: 
            R_v = 3.1
        xc=(1./wav) * 1.0e-6
        #For a(x)
        if 0.3 < xc < 1.1:
            a=0.574 * (xc**1.61)
        elif 1.1 < xc < 3.3:
            a=(1 + 0.17699 * (xc-1.82) - 0.50447 * (xc-1.82)**2 - 0.02427 * (xc-1.82)**3 + 0.72085 * 
            (xc-1.82)**4 + 0.01979 * (xc-1.82)**5 - 0.77530 * (xc-1.82)**6 + 0.32999 * (xc-1.82)**7)
        elif 3.3 < xc < 8.:
            a=1.752 - 0.316 * xc - 0.104/((xc - 4.67)**2 + 0.341) + self.F_a(xc)
        elif 8.0 < xc <10.:
            a=-1.073 -0.628*(xc-8) +0.137*(xc-8)**2 - 0.07*(xc-8)**3

        #For b(x)
        if 0.3 < xc < 1.1:
            b=-0.527 * (xc**1.61)
        elif 1.1 < xc <3.3:
            b=(1.41338*(xc-1.82) + 2.28305 * (xc-1.82)**2 + 1.07233 * (xc-1.82)**3 - 5.38434 * 
                (xc-1.82)**4 - 0.62251 * (xc-1.82)**5 + 5.30260 * 
                (xc-1.82)**6 - 2.09002*(xc-1.82)**7)
        elif 3.3 < xc < 8.:
            b=-3.090 + 1.825 * xc + 1.206/((xc - 4.62)**2 +0.263) + self.F_b(xc)
        elif 8. < xc <10.:
            b=13.67 + 4.257*(xc-8) -0.42*(xc-8)**2 +0.374*(xc-8)**3

        return a + b/R_v

    @staticmethod
    @jit(nopython=True, cache=True)
    def smc(wav, R_v=None):
        if R_v:
            R_v = R_v
        else: 
            R_v = 1.475
        xc = (1/wav)*10**-6
        D = xc**2/((xc**2 - 4.558**2)**2+xc**2*0.945**2)
        if xc < 5.9: F = 0
        else: F = 0.5392*(xc-5.9)**2 + 0.05644*(xc-5.9)**2
        b = -1.475 + 1.132*xc + 1.463*D + 0.294*F
        return 1 + b/R_v

