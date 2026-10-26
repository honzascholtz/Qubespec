import numpy as np
from astropy.modeling.powerlaws import PowerLaw1D

class general_model:
    """
    Generic gaussian-lines + continuum model, built from a `components` dict instead of being
    hand-written per line set.

    ``general_model`` inspects ``components`` and works out the free parameters it needs
    (``self.labels``), so you only have to supply a matching ``priors`` entry for each of them -
    see the "Fitting a custom model with ``general_model``" section of the docs for the full
    ``components``/``continuum`` dictionary format and a worked example.

    Parameters
    ----------
    components : dict
        Maps a component name to a dict describing that emission line: ``'wave'`` (rest-frame
        wavelength in Angstrom), ``'z'`` (name of the shared redshift parameter), ``'fwhm'``
        (name of the shared FWHM parameter), and optionally ``'ratio_to'``/``'ratio'`` (to tie
        this line's amplitude to another component's, e.g. for a fixed-ratio doublet) or
        ``'peak_name'`` (to override the default ``f'{name}_peak'`` amplitude parameter name).
        An optional ``'continuum'`` entry configures the continuum instead of describing a line
        (keys ``'type'`` - ``'power'``/``'linear'``/``'none'``, plus optional ``'wave'``/``'z'``
        pivot overrides).
    priors : dict
        Full priors dict (same format as elsewhere: ``{'param': [init, dist, *dist_args]}``).
        Passed straight through and used as given - this class does not invent priors, it only
        checks that every parameter in ``self.labels`` has an entry.
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

    def gauss(self, x, k, mu, FWHM):
        sig = FWHM/3e5*mu/2.35482
        expo = -((x-mu)**2)/(2*sig*sig)
        return k*np.exp(expo)

    def model(self, x, *pars):
        p = dict(zip(self.labels, pars))

        amp = {}
        for name, c in self.components.items():
            if 'ratio_to' not in c:
                amp[name] = p[c.get('peak_name', f'{name}_peak')]
        for name, c in self.components.items():
            if 'ratio_to' in c:
                amp[name] = c['ratio']*amp[c['ratio_to']]

        self.profiles = {}
        self.lines = 0.
        for name, c in self.components.items():
            mu = c['wave']*(1 + p[c['z']])/1e4
            self.profiles[name] = self.gauss(x, amp[name], mu, p[c['fwhm']])
            self.lines = self.lines + self.profiles[name]

        if self.continuum == 'power':
            pivot = self.cont_wave/1e4*(1 + p[self.cont_z])
            self.cont = PowerLaw1D.evaluate(x, p['cont'], pivot, alpha=p['cont_grad'])
        elif self.continuum == 'linear':
            self.cont = p['cont_grad']*x + p['cont']
        else:
            self.cont = 0.

        return self.cont + self.lines

    
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