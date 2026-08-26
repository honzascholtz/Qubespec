import numpy as np
import numba

_CONT_POWER, _CONT_LINEAR, _CONT_NONE = 0, 1, 2


@numba.njit
def _evaluate_general_model(x, pars, rest_wave, z_idx, fwhm_idx, peak_idx, ratio_coef,
                             cont_type, cont_idx, cont_grad_idx, cont_z_idx, cont_pivot_wave):
    lines = np.zeros_like(x)
    for i in range(rest_wave.shape[0]):
        mu = rest_wave[i]*(1 + pars[z_idx[i]])/1e4
        sig = pars[fwhm_idx[i]]/3e5*mu/2.35482
        amp = ratio_coef[i]*pars[peak_idx[i]]
        lines += amp*np.exp(-((x - mu)**2)/(2*sig*sig))

    if cont_type == _CONT_POWER:
        pivot = cont_pivot_wave/1e4*(1 + pars[cont_z_idx])
        cont = pars[cont_idx]*(x/pivot)**(-pars[cont_grad_idx])
    elif cont_type == _CONT_LINEAR:
        cont = pars[cont_grad_idx]*x + pars[cont_idx]
    else:
        cont = np.zeros_like(x)

    return cont + lines


class general_model:
    """
    Generic gaussian-lines + power-law-continuum model, built from a
    `components` dict instead of being hand-written per line set (c.f. HeII_model above).

    components : dict
        name -> {
            'wave'      : rest-frame wavelength in Angstrom,
            'z'         : name of the redshift parameter this line's centroid uses
                          (components sharing the same 'z' string share one fitted redshift),
            'fwhm'      : name of the FWHM parameter this line uses
                          (components sharing the same 'fwhm' string share one fitted FWHM),
            'ratio_to'  : (optional) name of another component this one's amplitude is tied to,
            'ratio'     : (optional, required with 'ratio_to') fixed flux ratio vs that component,
            'peak_name' : (optional) override for the free amplitude parameter's name,
                          default is f'{name}_peak'. Ignored if 'ratio_to' is set.
        }
        An optional 'continuum' entry configures the continuum instead of being a line:
        'continuum': {
            'type' : {'power', 'linear', 'none'}, optional, default 'power',
            'wave' : rest wavelength (Angstrom) used as the power-law/linear pivot,
                     default is the wavelength of the first line component,
            'z'    : name of the z-group used for that pivot,
                     default is the 'z' of the first line component,
        }
    priors : dict
        Full priors dict (same format as elsewhere: {'param': [init, dist, *dist_args]}).
        Passed straight through and used as given - this class does not invent priors,
        it only checks that every parameter in self.labels has an entry.
    """
    def __init__(self, components, priors):
        components = dict(components)
        cont_cfg = components.pop('continuum', {})
        self.continuum = cont_cfg.get('type', 'power')
        if self.continuum not in ('power', 'linear', 'none'):
            raise ValueError("continuum 'type' must be one of 'power', 'linear', 'none'")
        self.components = components

        first = next(iter(components.values()))
        self.cont_wave = cont_cfg.get('wave', first['wave'])
        self.cont_z = cont_cfg.get('z', first['z'])

        z_names = list(dict.fromkeys(c['z'] for c in components.values()))
        fwhm_names = list(dict.fromkeys(c['fwhm'] for c in components.values()))
        peak_names = [c.get('peak_name', f'{name}_peak')
                      for name, c in components.items() if 'ratio_to' not in c]

        cont_labels = [] if self.continuum == 'none' else ['cont', 'cont_grad']
        self.labels = cont_labels + z_names + fwhm_names + peak_names

        missing = [lab for lab in self.labels if lab not in priors]
        if missing:
            raise KeyError(f'priors is missing entries for: {missing}')
        self.priors = priors

        # Flatten components into plain numpy arrays so `model()` can be evaluated
        # by a numba-jitted function - fast enough to call once per emcee step.
        idx = {lab: i for i, lab in enumerate(self.labels)}
        self._rest_wave = np.array([c['wave'] for c in components.values()])
        self._z_idx = np.array([idx[c['z']] for c in components.values()])
        self._fwhm_idx = np.array([idx[c['fwhm']] for c in components.values()])
        self._peak_idx = np.array([
            idx[components[c['ratio_to']].get('peak_name', f"{c['ratio_to']}_peak")]
            if 'ratio_to' in c else idx[c.get('peak_name', f'{name}_peak')]
            for name, c in components.items()
        ])
        self._ratio_coef = np.array([c.get('ratio', 1.0) for c in components.values()])

        self._cont_type_code = {'power': _CONT_POWER, 'linear': _CONT_LINEAR, 'none': _CONT_NONE}[self.continuum]
        self._cont_idx = idx['cont'] if self.continuum != 'none' else -1
        self._cont_grad_idx = idx['cont_grad'] if self.continuum != 'none' else -1
        self._cont_z_idx = idx[self.cont_z] if self.continuum == 'power' else -1

    def model(self, x, *pars):
        pars = np.asarray(pars, dtype=np.float64)
        return _evaluate_general_model(
            x, pars, self._rest_wave, self._z_idx, self._fwhm_idx, self._peak_idx, self._ratio_coef,
            self._cont_type_code, self._cont_idx, self._cont_grad_idx, self._cont_z_idx, self.cont_wave,
        )

    def line_profiles(self, x, *pars):
        """Slow, diagnostic per-line breakdown (e.g. for plotting) - not used by the fit itself."""
        p = dict(zip(self.labels, pars))

        amp = {}
        for name, c in self.components.items():
            if 'ratio_to' not in c:
                amp[name] = p[c.get('peak_name', f'{name}_peak')]
        for name, c in self.components.items():
            if 'ratio_to' in c:
                amp[name] = c['ratio']*amp[c['ratio_to']]

        profiles = {}
        for name, c in self.components.items():
            mu = c['wave']*(1 + p[c['z']])/1e4
            sig = p[c['fwhm']]/3e5*mu/2.35482
            profiles[name] = amp[name]*np.exp(-((x - mu)**2)/(2*sig*sig))

        return profiles

    
if __name__ == '__main__':
    z = 10.602
    components_narrow = {
    'HeII1640': {'wave': 1640.420, 'z': 'z', 'fwhm': 'FWHM'},
    'OIII1660': {'wave': 1660.809, 'z': 'z', 'fwhm': 'FWHM', 'ratio_to': 'OIII1666', 'ratio': 1/3},
    'OIII1666': {'wave': 1666.150, 'z': 'z', 'fwhm': 'FWHM', 'peak_name': 'OIII1663_peak'},
    'continuum': {'type': 'power', 'wave': 1640.420, 'z': 'z'},
    }

    components_broad = dict(components_narrow)
    components_broad['HeII1640_broad'] = {
        'wave': 1640.420, 'z': 'zBLR', 'fwhm': 'FWHM_broad', 'peak_name': 'HeII1640_peak_broad',
    }

    priors = {}
    priors['z'] = [z, 'normal_hat', z, 0.01, z-0.06, z+0.06]
    priors['zBLR'] = [z, 'normal_hat', z, 0.01, z-0.06, z+0.06]
    priors['cont'] = [2, 'uniform', 0, 4]
    priors['cont_grad'] = [-2, 'normal', -2, 0.6]
    priors['HeII1640_peak'] = [1.3, 'loguniform', -3, 1]
    priors['HeII1640_peak_broad'] = [0.3, 'loguniform', -3, 1]
    priors['OIII1663_peak'] = [1.3, 'loguniform', -3, 1]
    priors['FWHM'] = [300, 'uniform', 200, 600]
    priors['FWHM_broad'] = [800, 'uniform', 600, 10000]

    gal_mod = general_model(components_narrow, priors)
    blr_mod = general_model(components_broad, priors)

    print('narrow labels:', gal_mod.labels)
    print('broad labels: ', blr_mod.labels)