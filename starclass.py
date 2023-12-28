from zipfile import ZipFile, ZIP_DEFLATED
from modules.RLC import RLC
import pandas as pd
import numpy as np
import os, json, traceback, time

# NGC 6819 TFA: TFA-20211224T034608Z-001.zip
# NGC 6791 TFA: TFA-20210902T000721Z-001.zip

cl={'ngc6791':{'rlcs':'TFA_q05_ngc6791.zip', 'catalog':'catalog_with_astrobase_periods_addon_q05.csv'}, 
    'ngc6819':{'rlcs':'TFA_q05_ngc6819.zip', 'catalog':'Bauer_Classified_Test_Set_NGC6819.csv'}}

class Star():
    '''Class - Star:
        Star.ID (dtype: str) Gaia ID,
        Star.ra (dtype: float) Right-ascension (decimal),
        Star.dec (dtype: float) Declination (decimal),
        Star.Gaia_mag (dtype: float) Average Gaia magnitude,
        Star.BP_mag (dtype: float) Average Vega BP magnitude,
        Star.color_BP_RP (dtype: float) Color: BP-RP,
        Star.color_BP_G (dtype: float) Color: BP-G,
        Star.color_G_RP (dtype: float) Color: G-RP,
        Star.RLC (dtype: RLC object) RLC object
        Star.inrad (dtype: generator object) Call next(Star.inrad) to generate a (self-including) list of all neighbor Gaia IDs within a 0.0125 degree radius of the current star,
        
        Star.export_data() (Method: returns dict (if dump=False) or exports as .json file) Compiles the available data of the star's photometric data to a .json file or returns a dictionary of available data.'''
    def __init__(self, Qdfs, ID, cluster='ngc6819'):
        self.ID=ID
        self.ra=catalog.at[ID, 'ra']
        self.dec=catalog.at[ID, 'dec']
        self.Gaia_mag=catalog.at[ID, 'Gaia_mag']
        self.BP_mag=catalog.at[ID, 'BP_mag']
        self.color_BP_RP=catalog.at[ID, 'bp_rp']
        self.color_BP_G=catalog.at[ID, 'bp_g']
        self.color_G_RP=catalog.at[ID, 'g_rp']
        self.inrad=self.__inrad__()
        self.RLC=RLC(ID, cluster=cluster)
        
    def __repr__(self):
        return f"Star - RA, Dec, Gaia mag, BP mag, BP-RP, BP-G, G-RP, inrad*, RLC: {'Available' if not isinstance(self.RLC.time, type(None)) else 'Not found'}"

    def __str__(self):
        return f"Star({self.ID}) - RA={round(self.ra, 2):6}, Dec={round(self.dec, 3):6}, Gaia mag={round(self.Gaia_mag, 2):5}, BP mag={round(self.BP_mag, 2):5}, BP-RP={round(self.color_BP_RP, 2):3}, BP-RP={round(self.color_BP_G, 2):3}, G-RP={round(self.color_G_RP, 2):3}, RLC: {'Available' if not isinstance(self.RLC.time, type(None)) else 'Not found'}"
    
    def __inrad__(self, radius=.0125):
        inrad=catalog[['ra','dec']].copy()
        inrad['hypot']=np.sqrt((inrad.dec-self.dec)**2+(inrad.ra-self.ra)**2)
        inrad=inrad.sort_values('hypot')
        while True:
            yield inrad.index[inrad['hypot']<=radius]
    
    def export_features(self, cluster, pos_data=True, mags=True, colors=True, neighbors=False, nonper_data=True, dump=True, per_period=False, truncate=True, verbose=True):
        print(f"Exporting features for Gaia {self.ID}") if verbose else None
        star_data={'ID':self.ID}
        if pos_data:
            star_data={**star_data, 'ra':self.ra, 'dec':self.dec}
        if mags:
            star_data={**star_data, 'Gaia_mag':self.Gaia_mag, 'BP_mag':self.BP_mag}
        if colors:
            star_data={**star_data, 'BP_RP':self.color_BP_RP, 'BP_G':self.color_BP_G, 'G_RP':self.color_G_RP}
        if neighbors:
            star_data={**star_data, 'inrad':next(self.inrad)}
        if not isinstance(self.RLC.time, type(None)):
            def try_export(cluster, star_data, pgram, tag, i):
                pgram_tags={0:'BLS', 1:'GLS', 2:'Stwf'}
                version={0:'a', 1:'b', 2:'c'}
                file_id=f"{star_data['ID']}_{pgram_tags[tag]}_{version[i]}.json"
                try:
                    print("Creating file: ", os.path.join(f'features_{cluster}', file_id))
                    export_period={**star_data, **{'lspval':pgram['nbestlspvals'][i]}, **self.RLC.analyze_period(pgram['nbestperiods'][i])}
                    with open(os.path.join(f'features_{cluster}', f'{self.ID}_{pgram_tags[tag]}_{version[i]}.json'), 'w+') as f:
                        json.dump(export_period, f, indent=4)
                except Exception as exc:
                    print(f"Error exporting {self.ID}_{pgram_tags[tag]}_{version[i]}.json") if verbose else None
                    print(traceback.format_exc())
            if nonper_data:
                star_data={**star_data, **self.RLC.nonperiodic_feats}
                for tag, pgram in enumerate([self.RLC.BLS, self.RLC.GLS, self.RLC.Stwf]):
                    if per_period and dump:
                        for i in range(3):
                            try:
                                if all(pgram['siderealflags']) or not pgram['siderealflags'][i]:
                                    args=[cluster, star_data, pgram, tag, i]
                                    try_export(*args)
                            except:
                                print("Unknown error(s) encountered in file generation procedure. Aborting export.")
                                continue
            elif per_period and not dump:
                raise NotImplementedError
            elif dump and not per_period:
                for i in range(3):
                    star_data={**star_data, **{"lspval":pgram['nbestlspvals'][i]}, **self.RLC.analyze_period(pgram['nbestperiods'][i])}
                    with open(os.path.join('features', f'{self.ID}.json'), 'w') as f:
                        json.dump(star_data, f, indent=4)
            else:
                return json.dumps(star_data, indent=4)
        else:
            print("RLC does not contain data.")

def load_catalog(cluster):
    global catalog
    catalog=pd.read_csv(f"astrobase_out/{cl[cluster]['catalog']}") # Change necessary pathways
    catalog=catalog[['GaiaID', 'ra', 'dec', 'Gaia_mag', 'BP_mag', 'bp_rp', 'bp_g', 'g_rp']].astype({'GaiaID':str})
    catalog=catalog[catalog['Gaia_mag']<=20.0].set_index('GaiaID')
    return catalog

def MakeStardict(cluster='ngc6819', must_not_exist=False, sel=None, export_all=False, truncate=True, verbose=True):
    '''Loads defined quarters for use in star object initialization. Paths will need to be edited in the previous function. If sel=None, will create stardict with ALL stars and will run the corresponding RLC commands (periodograms, analysis, fits) on the whole catalog--I HIGHLY recommend providing a list of the Gaia ID's to go through only a few at a time!'''
    global catalog
    to=int(time.time()) 
    catalog=load_catalog(cluster)
    if export_all:
        ti=int(time.time())
        available={rlc[4:-4] for rlc in ZipFile(os.path.join('TFA_LCS', cl[cluster]['rlcs']), compression=ZIP_DEFLATED).namelist() if not rlc.endswith(".dat")}
        allstars=catalog[catalog[['ra', 'dec', 'Gaia_mag']].index.isin(available)]
        if must_not_exist:
            existing_files={file[:19] for file in os.listdir(os.path.join(f'features_{cluster}'))}
            not_generated={star for star in allstars.index.to_list() if not star in existing_files}
            print(f"Available for analysis: {len(allstars)} stars. Exporting {len(not_generated)} stars.")
            allstars=allstars[~allstars.index.isin(existing_files)]
        for completed, ID in enumerate(allstars.index.to_list()):
            print(f"\nGenerating feature set for Gaia {ID}.")
            try:
                Star([catalog], ID, cluster).export_features(per_period=True, truncate=truncate, verbose=verbose, cluster=cluster)
                print(f"Features exported. {completed+1}/{len(allstars)} - {round(100*((completed+1)/len(allstars)), 3)}% - Rate {round((time.time()-ti)/60/(completed+1), 3)} mins/star - MET: {round((time.time()-to)/60**2, 2)} hrs") if verbose else None
            except KeyError:
                with open(os.path.join(f"features_{cluster}", "errs.txt"), 'a') as errs:
                    errs.write(ID)
                    print(f"Error keying Gaia {ID}. Cancelling data export of {ID}.json") if verbose else None
            except AttributeError:
                print("AttributeError encountered. Cancelling star export") if verbose else None
                with open(os.path.join(f"features_{cluster}", "errs.txt"), 'a') as errs:
                    errs.write(ID)
                continue
    elif not type(sel)==list:
        stars=catalog[['ra', 'dec', 'Gaia_mag']]
        stardict=dict(zip(stars.index, [Star([catalog], ID) for ID in stars.index]))
    else:
        stardict=dict(zip(sel, [Star([catalog], ID) for ID in sel]))
#     return stardict