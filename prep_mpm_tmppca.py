import nibabel as nib
import numpy as np
from  tmppca import denoise_tmppca_cli

import logging, os, re, json, sys, enum, pathlib, shutil


#%% Config

main_dir = pathlib.Path('/network/iss/cenir/analyse/irm/users/benoit.beranger/2026_07_30_DEV2_444_09_compareMPM')
work_dir = main_dir / 'mpm_tmppca_good'

regex_PD = re.compile(r'.*_PDw_good$')
regex_MT = re.compile(r'.*_MTw_good$')
regex_T1 = re.compile(r'.*_T1w_good$')

regex_rawnifti  = re.compile(r"^v_.*nii$")


#%% Setup

logging.basicConfig(
    level=logging.DEBUG,
    format=f"%(levelname)8s: %(message)s",
)
logger = logging.getLogger()
logger.info(f'main_dir: {main_dir}')


class File:

    def __init__(self) -> None:
        self.is_mag         : bool          = None
        self.contrast       : Contrast      = None
        self.echo_number    : int           = None
        self.echo_time      : float         = None
        self.info           : dict          = None

    def __repr__(self) -> str:
        return str(self.__dict__)

    def __lt__(self, other) -> bool:
        return  self.echo_time < other.echo_time
    
    def __bool__(self) -> bool:
        return len(self.nii)>0
    
    @property
    def main_dir(self):
        return self._main_dir
    @property
    def root(self):
        return self._root
    @property
    def raw_nii(self):
        return self._raw_nii
    @property
    def raw_json(self):
        return self._raw_json
    @property
    def den_nii(self):
        return self._den_nii
    @property
    def den_json(self):
        return self._den_json
    
    @main_dir.setter
    def main_dir(self, value):
        self._main_dir = pathlib.Path(value)
    @root.setter
    def root(self, value):
        self._root = pathlib.Path(value)
    @raw_nii.setter
    def raw_nii(self, value):
        self._raw_nii = pathlib.Path(value)
    @raw_json.setter
    def raw_json(self, value):
        self._raw_json = pathlib.Path(value)
    @den_nii.setter
    def den_nii(self, value):
        self._den_nii = pathlib.Path(value)
    @den_json.setter
    def den_json(self, value):
        self._den_json = pathlib.Path(value)

class Contrast(enum.StrEnum):
    PD = 'PD'
    MT = 'MT'
    T1 = 'T1'
    

#%% Fetch contrast dirs files

raw_sources: list[File] = []

for root, dirs, files in os.walk(main_dir):
    for file in files:
        if regex_rawnifti.match(file):
            new = File()
            new.main_dir = main_dir
            new.root     = root
            new.raw_nii  = new.root / file
            jsonfile     = new.raw_nii.with_suffix('.json')
            if jsonfile.is_file():
                new.raw_json = jsonfile
                with open(jsonfile,'r') as fid:
                    new.info = json.load(fid)
            if regex_PD.match(root) or regex_MT.match(root) or regex_T1.match(root):
                if   new.info['ImageType'][2] == 'M': new.is_mag = True
                elif new.info['ImageType'][2] == 'P': new.is_mag = False
                new.echo_number = new.info['EchoNumber']
                new.echo_time   = new.info['EchoTime']
                if regex_PD.match(root): new.contrast = Contrast.PD
                if regex_MT.match(root): new.contrast = Contrast.MT
                if regex_T1.match(root): new.contrast = Contrast.T1
                new.den_nii = work_dir / f'den_{new.contrast}_{new.echo_number}.nii'
                new.den_json = new.den_nii.with_suffix('.json')
                raw_sources.append(new)

raw_mag = list(filter(lambda x: x.is_mag==True, raw_sources))

raw_PD = list(filter(lambda x: x.contrast==Contrast.PD, raw_mag))
raw_MT = list(filter(lambda x: x.contrast==Contrast.MT, raw_mag))
raw_T1 = list(filter(lambda x: x.contrast==Contrast.T1, raw_mag))

raw_PD = sorted(raw_PD)
raw_MT = sorted(raw_MT)
raw_T1 = sorted(raw_T1)

raw_MPM = raw_PD + raw_MT + raw_T1

n_PD = len(raw_PD)
n_MT = len(raw_MT)
n_T1 = len(raw_T1)

if n_PD < 1: raise FileNotFoundError(f'no raw_PD found with {main_dir / regex_PD.pattern / regex_rawnifti.pattern}')
if n_MT < 1: raise FileNotFoundError(f'no raw_MT found with {main_dir / regex_MT.pattern / regex_rawnifti.pattern}')
if n_T1 < 1: raise FileNotFoundError(f'no raw_T1 found with {main_dir / regex_T1.pattern / regex_rawnifti.pattern}')
logger.info(f'found n_PD={n_PD} raw_PD magnitudes')
logger.info(f'found n_MT={n_MT} raw_MT magnitudes')
logger.info(f'found n_T1={n_T1} raw_T1 magnitudes')

logger.info(f'PD // sorted TE = {[round(file.echo_time*1000,3) for file in raw_PD]}(ms)')
logger.info(f'MT // sorted TE = {[round(file.echo_time*1000,3) for file in raw_MT]}(ms)')
logger.info(f'T1 // sorted TE = {[round(file.echo_time*1000,3) for file in raw_T1]}(ms)')


#%% Copy data in work dir using symlinks

work_dir.mkdir(parents=True, exist_ok=True)
logger.info(f'work_dir is : {work_dir}')


#%% Prepare MPM images : concatenate all raw images, for easy visual QC

logger.info(f'Loading header raw_PD'); img_PD_raw = [nib.load(file.root / file.raw_nii) for file in raw_PD]
logger.info(f'Loading header raw_MT'); img_MT_raw = [nib.load(file.root / file.raw_nii) for file in raw_MT]
logger.info(f'Loading header raw_T1'); img_T1_raw = [nib.load(file.root / file.raw_nii) for file in raw_T1]

logger.info(f'Loading data raw_PD'); data_PD_raw = np.stack([img.get_fdata() for img in img_PD_raw], axis=3)
logger.info(f'Loading data raw_MT'); data_T1_raw = np.stack([img.get_fdata() for img in img_T1_raw], axis=3)
logger.info(f'Loading data raw_T1'); data_MT_raw = np.stack([img.get_fdata() for img in img_MT_raw], axis=3)

logger.info(f'Stacking raw images into single 4D')
data_MPM_raw = np.concat((data_PD_raw, data_MT_raw, data_T1_raw), axis=3)
logger.info(f'data_raw shape is {data_MPM_raw.shape}')

path_MPM_raw = work_dir / 'data_MPM_raw.nii'
img_raw = nib.Nifti1Image(
    dataobj=data_MPM_raw,
    affine=img_PD_raw[0].affine,
    header=img_PD_raw[0].header)
img_raw.header['dim'][0] = 4
logger.info(f'Writing raw 4D images in disk: {path_MPM_raw}'); nib.save(img_raw, path_MPM_raw)


#%% Run tMPPCA

path_MPM_den = work_dir / 'data_MPM_den.nii'
sys.argv = [
    'denoise-tmppca',
    '--sigma2'  , str(work_dir /   'sigma2.nii'),
    '--snr_gain', str(work_dir / 'snr_gain.nii'),
    '--window', '5,5,5',
    str(path_MPM_raw),
    str(path_MPM_den)
]
denoise_tmppca_cli.main()


#%% Deconcatenate MPM denoised data

logger.info(f'Loading denoised images: {path_MPM_den}');
img_MPM_den = nib.load(path_MPM_den)
data_MPM_den = img_MPM_den.get_fdata()

N = n_PD + n_MT + n_T1

for n in range(N):
    data_echo = data_MPM_den[:,:,:,n]
    img_den = nib.Nifti1Image(
        dataobj=data_echo,
        affine=img_MPM_den.affine,
        header=img_MPM_den.header)
    logger.info(f'Writing {raw_MPM[n].den_nii.stem}[.nii, .json]')
    nib.save(img_den, raw_MPM[n].den_nii)
    shutil.copyfile(raw_MPM[n].raw_json,raw_MPM[n].den_json)

logger.info(f'All done')

