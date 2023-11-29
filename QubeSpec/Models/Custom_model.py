import numpy as np

#Dependencies
import numpy as np
import emcee
import scipy.stats as stats
from multiprocessing import Pool
from astropy.modeling.powerlaws import PowerLaw1D

#Imports needed for testing
from astropy.io import fits
import matplotlib.pyplot as plt


#Parameter class for storing priors
class Parameter:
    def __init__(self, value, name, prior):
        self.value = value
        self.name = name 
        self.prior_params = prior
    def log_prior(self):
        match self.prior_params[0]:
            case "uniform": return stats.uniform.logpdf(self.value, self.prior_params[1], self.prior_params[2]-self.prior_params[1])
            case 'loguniform': return stats.uniform.logpdf(np.log10(self.value), self.prior_params[1], self.prior_params[2]-self.prior_params[1])
            case 'normal': return stats.norm.logpdf(self.value, self.prior_params[1], self.prior_params[2])
            case'lognormal':
                return stats.lognorm.logpdf(self.value, self.prior_params[1], self.prior_params[2])
            case 'normal_hat':
                return stats.truncnorm.logpdf(self.value, self.prior_params[1], self.prior_params[2], self.prior_params[3],self.prior_params[4])
            case _:
                raise NameError("Prior {} not found".format(self.prior_params[0]))
    
    def sample_prior(self, N):
        match self.prior_params[0]:
            case "uniform": return stats.uniform.rvs( self.prior_params[1], self.prior_params[2]-self.prior_params[1], N )
            case 'loguniform': return 10**stats.uniform.rvs( self.prior_params[1], self.prior_params[2]-self.prior_params[1],N)
            case 'normal': return stats.norm.rvs( self.prior_params[1], self.prior_params[2], N)
            case'lognormal':
                return stats.lognorm.logpdf( self.prior_params[1], self.prior_params[2], N)
            case 'normal_hat':
                return stats.truncnorm.logpdf(self.prior_params[1], self.prior_params[2], self.prior_params[3],self.prior_params[4],N)
            case _:
                raise NameError("Prior {} not found".format(self.prior_params[0]))

###########Line models
class LineModel:
    def __init__(self, name, parameters, rest_wav, width_type=""):
        self.name = name
        self.rest_wav = rest_wav
        self.parameters = parameters
        self.width_type = width_type

    def gauss(self, x, k, mu, sig):
        expo = -((x-mu)**2)/(2*sig*sig)
        y = k * np.e**expo
        return y

    def fwhm_conv(self, fwhm_in, central_wav):
        return (fwhm_in/2.355)*central_wav/(3*10**5)

    def return_value(self, in_wavelenght):
        #print(self.rest_wav)
        cen_wav = self.rest_wav*(1+self.parameters[0])
        sigma = self.fwhm_conv(self.parameters[2], cen_wav)
        return self.gauss(in_wavelenght, self.parameters[1], cen_wav, sigma)

class DoubletModel:
    def __init__(self, name, parameters, rest_wav1, rest_wav2, width_type=""):
        self.name = name
        self.rest_wav1 = rest_wav1
        self.rest_wav2 = rest_wav2
        self.parameters = parameters
        self.width_type = width_type
        
    def gauss(self, x, k, mu, sig):
        expo = -((x-mu)**2)/(2*sig*sig)
        y = k * np.e**expo
        return y

    def fwhm_conv(self, fwhm_in, central_wav):
        return (fwhm_in/2.355)*central_wav/(3*10**5)

    def return_value(self, in_wavelength):
        peak1 = self.parameters[1]
        peak2 = peak1/self.parameters[3]

        cen_wav1 = self.rest_wav1*(1+self.parameters[0])
        sigma1 = self.fwhm_conv(self.parameters[2], cen_wav1)
        cen_wav2 = self.rest_wav2*(1+self.parameters[0])
        sigma2 = self.fwhm_conv(self.parameters[2], cen_wav2)
        flux = self.gauss(in_wavelength, peak1, cen_wav1, sigma1)+\
            self.gauss(in_wavelength, peak2, cen_wav2, sigma2)
        return flux

##############Generic 'build your own line' class:
class Model:
    def __init__(self, model_name, input_parameters):
        self.model_name = model_name #Enter model name
        self.lines = {}
        self.theta = {}
        #input_parameters key format: purpose_narrow/broad_name_type
        line_parameters = {}
        doublet_parameters = {}
        ratio_parameters = {}
        for key in input_parameters.keys():
            split_key = key.split('_')
            value = input_parameters[key][0]
            purpose = split_key[0]
            name = split_key[1]
            #Load a common model parameter
            match purpose:
                case 'm':
                    if len(split_key)==2: self.theta[name] = Parameter(value, name, input_parameters[key][1])
                    else:
                        self.theta[name+"_"+ split_key[2]] = Parameter(value, name +"_"+split_key[2], 
                        input_parameters[key][1])

            #Load line parameter
                case 'l':
                    name = split_key[2]
                    if name not in line_parameters: line_parameters[name] = [0, 0, 0, 0, 0]
                    param_name = split_key[3]
                    if len(input_parameters[key]) > 1: self.theta[name+'_'+param_name] = Parameter(value, name+'_'+param_name, 
                    input_parameters[key][1])
                    line_parameters[name][4] = split_key[1]
                    match param_name:
                        case 'z': line_parameters[name][0] = value
                        case 'peak': line_parameters[name][1] = value
                        case 'fwhm': line_parameters[name][2] = value
                        case 'wav': line_parameters[name][3] = value
            
            #Load doublet parameter
                case 'd':
                    name = split_key[2]
                    if name not in doublet_parameters: doublet_parameters[name] = [0, 0, 0, -1, -1, 0, 0, 0]
                    param_name = split_key[3]
                    doublet_parameters[name][7] = split_key[1]

                    if len(input_parameters[key]) > 1: 
                        self.theta[name+'_'+param_name] = Parameter(value, name+'_'+param_name, 
                    input_parameters[key][1])

                    match param_name:
                        case 'z': doublet_parameters[name][0] = value
                        case 'peak1': doublet_parameters[name][1] = value
                        case 'fwhm': doublet_parameters[name][2] = value
                        case 'ratio': doublet_parameters[name][3] = value
                        case 'peak2': doublet_parameters[name][4] = value
                        case 'wav1': doublet_parameters[name][5] = value
                        case 'wav2': doublet_parameters[name][6] = value
                
        #Initialize lines
        for line_name in line_parameters.keys():
            if line_parameters[line_name][0] == 0:
                line_parameters[line_name][0] = self.theta['z'].value

            if line_parameters[line_name][2] == 0:
                fwhm_type = line_parameters[line_name][4]
                line_parameters[line_name][2] = self.theta["fwhm_"+fwhm_type].value

            self.lines[line_name] = LineModel(line_name, line_parameters[line_name][:-2],
                line_parameters[line_name][-2], width_type=line_parameters[line_name][-1])    

        #Initialize doublets
        for doublet_name in doublet_parameters.keys():
            if doublet_parameters[doublet_name][0] == 0:
                doublet_parameters[doublet_name][0] = self.theta['z'].value

            if doublet_parameters[doublet_name][2] == 0:
                fwhm_type = doublet_parameters[doublet_name][7]
                doublet_parameters[doublet_name][2] = self.theta['fwhm_'+fwhm_type].value

            self.lines[doublet_name] = DoubletModel(doublet_name, doublet_parameters[doublet_name][:-3],
                doublet_parameters[doublet_name][-3], doublet_parameters[doublet_name][-2], width_type=doublet_parameters[doublet_name][-1])

    def prop_calc(self, results):  
        labels = list(results.keys())[1:]
        res_plt = []
        res_dict = {'name': results['name']}
        for lbl in labels:
        
            array = results[lbl]
        
            p50,p16,p84 = np.percentile(array, (50,16,84))
            p16 = p50-p16
            p84 = p84-p50
        
            res_plt.append(p50)
            res_dict[lbl] = np.array([p50,p16,p84])

        res_dict['popt'] = res_plt
        return res_dict

    def calculate_values(self, in_wavelength): #Return model values for plotting
        total = 0
        contm = 0
        for line in self.lines.values():
            total += line.return_value(in_wavelength)
        if ("ContSlope" and "ContNorm") in self.theta.keys():
            contm = PowerLaw1D.evaluate(in_wavelength, self.theta["ContNorm"].value,
            np.min(in_wavelength), alpha=self.theta["ContSlope"].value)
        return total+contm
    
    def update_parameters(self, new_params):
        for index, key in enumerate(self.theta.keys()):
            self.theta[key].value = new_params[index]
            if key == "ContNorm": self.theta[key]

        for key in self.theta.keys():
            split_key = key.split('_')
            if len(split_key) == 2 and split_key[0] != 'fwhm':
                name = split_key[0]
                value = self.theta[key].value
                #Update each line parameter
                match split_key[1]:
                    case 'z': 
                        self.lines[name].parameters[0] = value
                    case 'peak':
                        self.lines[name].parameters[1] = value
                    case 'peak1':
                        self.lines[name].parameters[1] = value
                    case 'peak2':    
                        self.lines[name].parameters[4] = value
                    case 'fwhm': self.lines[name].parameters[2] = value

                self.theta[key].value = value

            #Update fwhm if using global values
            for line in self.lines.values():
                fwhm_type = line.width_type
                if "fwhm_"+fwhm_type in self.theta.keys():
                    line.parameters[2] = self.theta["fwhm_"+fwhm_type].value

    #Evaluat
    def log_prior(self):
        logprior = 0
        for param in self.theta.values():
            logprior += param.log_prior()
        return logprior
    
    def log_prior_test(self):
        for param in self.theta.values():
            l= param.log_prior()
            if l==-np.inf:
                print('Prior returned infinity', param.value,param.prior_params[0] ,param.prior_params[0:])
                raise SyntaxError('Prior returned infinity - see above')

    #Chi2 log-likelihood
    def log_likelihood(self,):
        model = self.calculate_values(self.wave)
        sigma2 = self.error**2
        return -0.5*np.sum((self.flux-model)**2/sigma2)

    #Function to be called by mcmc
    def log_probability(self, theta):
        self.update_parameters(theta)
        lp = self.log_prior()
        if not np.isfinite(lp): return -np.inf
        else:
            return lp+self.log_likelihood()

    def fit_to_data(self, wave, flux, error, N=6000, nwalkers=32, ncpu=1, progress=True):
        self.wave = wave
        self.flux = flux
        self.error = error
        self.progress= progress
        self.ncpu = ncpu
        self.N = N 
        pos_l = np.array([par.value for par in self.theta.values()])
        ndim = len(pos_l)
        pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
        #pos[:,0] = np.random.normal(pos_l[0],0.001, nwalkers)
        
        for i in range(nwalkers):
            self.update_parameters(pos_l)
            self.log_prior_test()
        
        if self.ncpu==1:
            sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, self.log_probability, args=()) 
            
            sampler.run_mcmc(pos, self.N, progress=self.progress)
        
        elif self.ncpu>1:
            from multiprocess import Pool
            with Pool(self.ncpu) as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability, 
            args=(), pool = pool)

                sampler.run_mcmc(pos, self.N, progress=self.progress)

        #Extract chains
        self.flat_samples = sampler.get_chain(discard=int(0.25*N), thin=15, flat=True)
        self.labels = list(self.theta.keys())
        
        self.chains={'name':self.model_name}
        for i in range(len(self.labels)):
            self.chains[self.labels[i]] = self.flat_samples[:,i]
        self.props = self.prop_calc(self.chains) #Calculate properties
        self.update_parameters(self.props['popt']) #Set final parameters to best fit values
        
    