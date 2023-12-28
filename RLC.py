from zipfile import ZipFile, ZIP_DEFLATED
import pandas as pd
import numpy as np
from astrobase.varclass.periodicfeatures import *
from astrobase.varclass.varfeatures import *
from astrobase import periodbase
import os

cl={'ngc6791':{
        'rlcs':'TFA_q05_ngc6791.zip',
        'catalog':'catalog_with_astrobase_periods_addon_q05' 
},  'ngc6819':{
        'rlcs':'TFA_q05_ngc6819.zip', 
        'catalog':'Fully_Classifed_NGC6819'}
   }

class RLC:
    '''Class - RLC:
        RLC.time (dtype: np.ndarray or None) Array of time data,
        RLC.mag (dtype: np.ndarray or None) Array of magnitude data,
        RLC.err (dtype: np.ndarray or None) Array of magnitude error data.
        Note: If no RLC file is found, the following RLC attributes will NOT be defined:
        RLC.nonperiodic_feats (dtype: dict) Results of the astrobase nonperiodic features function,
        RLC.GLS (dtype: dict) Results of Lomb-Scargle periodogram,
        RLC.Stwf (dtype: dict) Results of Stellingwerf periodogram,
        RLC.BLS (dtype: dict) Results of box-least square periodogram,
        RLC.GLS_Stwf_feats (dtype: dict) Periodogram features of GLS and Stellingwerf algorithms (I couldn't find a way to resolve the value error from looking through documentation or otherwise),
        RLC.BLS_feats (dtype: dict) Periodogram features for BLS
        
        RLC.analyze_period(period) (Method - returns dict) Returns dictionary detailing phased and fitted LC features with respect to the provided period'''
    
    def __init__(self, ID, cluster='ngc6819', load_only=False):
        self.time, self.mag, self.err=self.__load_rlc__(ID, cluster=cluster)
        if not isinstance(self.time, type(None)) and not load_only: # Check the RLC has been successfully located and read, run analysis if true unless data is being loaded only (skip analysis)
            self.__analyze_rlc__()
            
    def __repr__(self):
        return self.__str__()
        
    def __str__(self):
        return f'RLC: {"Available" if not isinstance(self.time, type(None)) else "Not found"}'
        
    def __load_rlc__(self, ID, cluster='ngc6819'):
        '''The path to the archive of TFA data will need to be changed, and the archive of TFA data is expected to be a zipped directory to save memory'''
        try: # Read RLC from zip file & return time, mag, & err as numpy arrays for astrobase functions
            rlc=ZipFile(os.path.join('TFA_LCS', f"{cl[cluster]['rlcs']}"), compression=ZIP_DEFLATED).open(os.path.join('TFA', f'{ID}.rlc'), 'r')
            df=pd.read_csv(rlc, sep=' ', header=None)
            return (df[1].to_numpy(), df[6].to_numpy(), df[3].to_numpy())
        except FileNotFoundError:
            return (None, None, None)

    def __analyze_rlc__(self): # Assess nonperiodic features and run periodograms
        print(f"Running periodograms")
        self.nonperiodic_feats={key:value for (key, value) in nonperiodic_lightcurve_features(self.time, self.mag, self.err).items() if not type(value)==np.ndarray}
        self.GLS=periodbase.pgen_lsp(self.time, self.mag, self.err, startp=0.042, endp=20., stepsize=.001, sigclip=5., nbestpeaks=3, verbose=False)
        self.Stwf=periodbase.spdm.stellingwerf_pdm(self.time, self.mag, self.err, startp=.042, endp=20., stepsize=.001, sigclip=5., nbestpeaks=3, verbose=False)
        self.BLS=periodbase.bls_parallel_pfind(self.time, self.mag, self.err, startp=0.042, endp=20., stepsize=.001, sigclip=5., nbestpeaks=3, verbose=False)
        '''QUALITY CONTROL: Nyquist freq things and filter with pgram feats--this is not as straightforward as one would imagine, the period found doesn't have to be within a certain (small) fraction of Nyquist frequency multiples to have the appearance of holes in the data. Might be safer (for now) to omit all measurements of periods under a larger threshold. Cannot combine BLS with Stellingwerf and/or GLS to find periodogram features. Raises error and the documentation doesn't indicate why'''
        self.GLS_Stwf_feats=periodogram_features([self.GLS, self.Stwf], self.time, self.mag, self.err, sampling_startp=0.042, sampling_endp=20., sigclip=5, pdiff_threshold=0.005, sidereal_threshold=0.005, verbose=False)
        self.BLS_feats=periodogram_features([self.BLS], self.time, self.mag, self.err, sampling_startp=0.042, sampling_endp=20., sigclip=5., pdiff_threshold=0.005, sidereal_threshold=0.005, verbose=False)

    def analyze_period(self, period):
        '''ANALYZE PERIODS FROM PERIODOGRAMS: create list of dicts with info about the period and the fits for the folded data (kurt, skew, lcfit feats, etc.), tests with common period, for period in self.periods, append to list of dicts, self.period_dicts.append(self.analyze_periods(period))'''
        
        #### do i need to do a thing here?   \\//
        phased_feats=phasedlc_features(self.time, self.mag, self.err, period)
        fit_feats=lcfit_features(self.time, self.mag, self.err, period, sigclip=5., verbose=False)
        flat={'period':period, **{key:value for (key, value) in phased_feats.items() if not type(value)==np.ndarray}, **{key:value for (key, value) in fit_feats.items() if not type(value)==np.ndarray}}
        for sub in ['fourier_ampratios', 'fourier_phadiffs']:
            flat={**flat, **flat[sub]}
            del flat[sub]
        for tup in ['planet_residual_mad_over_lcmad', 'eb_residual_mad_over_lcmad', 'ebx2_residual_mad_over_lcmad']:
            flat[tup]=flat[tup][0]
        return flat