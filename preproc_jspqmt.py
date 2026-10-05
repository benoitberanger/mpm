#%% Imports

import nibabel as nib
import numpy as np

import logging, os, re, json, sys, enum, pathlib, shutil, subprocess


#%% Config

main_dir = pathlib.Path('/network/iss/cenir/analyse/irm/users/benoit.beranger/2026_07_30_DEV2_444_09_compareMPM')

# work_dir = main_dir / 'jspqmt_c9'
# regex_mt0 = re.compile(r'.*vibeMT_mt0_sag_6eco_1iso_c9$')
# regex_mtw = re.compile(r'.*vibeMT_mtw_sag_3eco_1iso_c9$')
# regex_pdw = re.compile(r'.*vibeMT_pdw_sag_6eco_1iso_c9$')
# regex_t1w = re.compile(r'.*vibeMT_t1w_sag_6eco_1iso_c9$')

# work_dir = main_dir / 'jspqmt_c9_es'
# regex_mt0 = re.compile(r'.*vibeMT_mt0_sag_6eco_1iso_c9_es$')
# regex_mtw = re.compile(r'.*vibeMT_mtw_sag_3eco_1iso_c9_es$')
# regex_pdw = re.compile(r'.*vibeMT_pdw_sag_6eco_1iso_c9_es$')
# regex_t1w = re.compile(r'.*vibeMT_t1w_sag_6eco_1iso_c9_es$')

work_dir = main_dir / 'jspqmt_c6'
regex_mt0 = re.compile(r'.*vibeMT_mt0_sag_6eco_1iso_c6$')
regex_mtw = re.compile(r'.*vibeMT_mtw_sag_3eco_1iso_c6$')
regex_pdw = re.compile(r'.*vibeMT_pdw_sag_6eco_1iso_c6$')
regex_t1w = re.compile(r'.*vibeMT_t1w_sag_6eco_1iso_c6$')

regex_rawnifti  = re.compile(r'^v_.*nii$')

regex_b1map = re.compile(r'.*b1map_3DREAM_relB1$')
b1map_factor: float = 1/1000


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
        self.is_b1map       : bool          = False
        self.contrast       : Contrast      = None
        self.echo_number    : int           = None
        self.echo_time      : float         = None
        self.info           : dict          = None

    def __repr__(self)        -> str : return str(self.__dict__)
    def __lt__  (self, other) -> bool: return self.echo_time < other.echo_time
    def __bool__(self)        -> bool: return len(self.nii)>0
    
    @property
    def main_dir(self): return self._main_dir
    @property
    def root    (self): return self._root
    @property
    def raw_nii (self): return self._raw_nii
    @property
    def raw_json(self): return self._raw_json
    @property
    def den_nii (self): return self._den_nii
    @property
    def den_json(self): return self._den_json
    
    @main_dir.setter
    def main_dir(self, value): self._main_dir = pathlib.Path(value)
    @root.setter
    def root    (self, value): self._root     = pathlib.Path(value)
    @raw_nii.setter
    def raw_nii (self, value): self._raw_nii  = pathlib.Path(value)
    @raw_json.setter
    def raw_json(self, value): self._raw_json = pathlib.Path(value)
    @den_nii.setter
    def den_nii (self, value): self._den_nii  = pathlib.Path(value)
    @den_json.setter
    def den_json(self, value): self._den_json = pathlib.Path(value)

class Contrast(enum.StrEnum):
    pdw = 'pdw'
    t1w = 't1w'
    mtw = 'mtw'
    mt0 = 'mt0'
    

#%% Prepare dirs

work_dir.mkdir(parents=True, exist_ok=True)
logger.info(f'work_dir is : {work_dir}')

denoising_dir = work_dir / '01_Denoising'
masking_dir   = work_dir / '02_Masking'
moco_dir      = work_dir / '03_MotionCorrection'
b1map_dir     = work_dir / '04_B1map'
denoising_dir.mkdir(parents=True, exist_ok=True)
masking_dir  .mkdir(parents=True, exist_ok=True)
moco_dir     .mkdir(parents=True, exist_ok=True)
b1map_dir    .mkdir(parents=True, exist_ok=True)

results_dir = work_dir / 'Results'
results_dir.mkdir(parents=True, exist_ok=True)


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
            if regex_mt0.match(root) or regex_mtw.match(root) or regex_pdw.match(root) or regex_t1w.match(root):
                if   new.info['ImageType'][2] == 'M': new.is_mag = True
                elif new.info['ImageType'][2] == 'P': new.is_mag = False
                new.echo_number = new.info['EchoNumber']
                new.echo_time   = new.info['EchoTime']
                if regex_mt0.match(root): new.contrast = Contrast.mt0
                if regex_mtw.match(root): new.contrast = Contrast.mtw
                if regex_pdw.match(root): new.contrast = Contrast.pdw
                if regex_t1w.match(root): new.contrast = Contrast.t1w
                new.den_nii = denoising_dir / f'den_{new.contrast}_e{new.echo_number}.nii'
                new.den_json = new.den_nii.with_suffix('.json')
                raw_sources.append(new)
            if regex_b1map.match(root):
                new.is_b1map = True
                raw_sources.append(new)

raw_mag = list(filter(lambda x: x.is_mag==True, raw_sources))

raw_mt0 = list(filter(lambda x: x.contrast==Contrast.mt0, raw_mag))
raw_mtw = list(filter(lambda x: x.contrast==Contrast.mtw, raw_mag))
raw_pdw = list(filter(lambda x: x.contrast==Contrast.pdw, raw_mag))
raw_t1w = list(filter(lambda x: x.contrast==Contrast.t1w, raw_mag))

raw_mt0 = sorted(raw_mt0)
raw_mtw = sorted(raw_mtw)
raw_pdw = sorted(raw_pdw)
raw_t1w = sorted(raw_t1w)

raw_qmt = raw_mt0 + raw_mtw + raw_pdw + raw_t1w

n_mt0 = len(raw_mt0)
n_mtw = len(raw_mtw)
n_pdw = len(raw_pdw)
n_t1w = len(raw_t1w)

if n_mt0 < 1: raise FileNotFoundError(f'no raw_mt0 found with {main_dir / regex_mt0.pattern / regex_rawnifti.pattern}')
if n_mtw < 1: raise FileNotFoundError(f'no raw_mtw found with {main_dir / regex_mtw.pattern / regex_rawnifti.pattern}')
if n_pdw < 1: raise FileNotFoundError(f'no raw_pdw found with {main_dir / regex_pdw.pattern / regex_rawnifti.pattern}')
if n_t1w < 1: raise FileNotFoundError(f'no raw_t1w found with {main_dir / regex_t1w.pattern / regex_rawnifti.pattern}')
logger.info(f'found n_mt0={n_mt0} raw_mtw magnitudes')
logger.info(f'found n_mtw={n_mtw} raw_mtw magnitudes')
logger.info(f'found n_pdw={n_pdw} raw_pdw magnitudes')
logger.info(f'found n_t1w={n_t1w} raw_t1w magnitudes')

logger.info(f'mt0 // sorted TE = {[round(file.echo_time*1000,3) for file in raw_mt0]}(ms)')
logger.info(f'mtw // sorted TE = {[round(file.echo_time*1000,3) for file in raw_mtw]}(ms)')
logger.info(f'pdw // sorted TE = {[round(file.echo_time*1000,3) for file in raw_pdw]}(ms)')
logger.info(f't1w // sorted TE = {[round(file.echo_time*1000,3) for file in raw_t1w]}(ms)')


raw_b1map: list[File] = list(filter(lambda x: x.is_b1map==True, raw_sources))
if len(raw_b1map) < 1: raise FileNotFoundError(f'no raw_b1map found with {main_dir / regex_b1map.pattern / regex_rawnifti.pattern}')
if len(raw_b1map) > 1: raise FileNotFoundError(f'too many raw_b1map found with {main_dir / regex_b1map.pattern / regex_rawnifti.pattern}')
raw_b1map: File = raw_b1map[0]


#%% Prepare qMT images : concatenate all raw images, for easy visual QC

logger.info(f'Loading header raw_mt0'); img_mt0_raw = [nib.load(file.root / file.raw_nii) for file in raw_mt0]
logger.info(f'Loading header raw_mtw'); img_mtw_raw = [nib.load(file.root / file.raw_nii) for file in raw_mtw]
logger.info(f'Loading header raw_pdw'); img_pdw_raw = [nib.load(file.root / file.raw_nii) for file in raw_pdw]
logger.info(f'Loading header raw_t1w'); img_t1w_raw = [nib.load(file.root / file.raw_nii) for file in raw_t1w]

logger.info(f'Loading data raw_mt0'); data_mt0_raw = np.stack([img.get_fdata() for img in img_mt0_raw], axis=3)
logger.info(f'Loading data raw_mtw'); data_mtw_raw = np.stack([img.get_fdata() for img in img_mtw_raw], axis=3)
logger.info(f'Loading data raw_pdw'); data_pdw_raw = np.stack([img.get_fdata() for img in img_pdw_raw], axis=3)
logger.info(f'Loading data raw_t1w'); data_t1w_raw = np.stack([img.get_fdata() for img in img_t1w_raw], axis=3)

path_qmt_raw = denoising_dir / '4D_raw.nii'
if path_qmt_raw.exists():
    logger.info(f'Loading raw 4D : {path_qmt_raw}')    
    img_qmt_raw = nib.load(path_qmt_raw)
    data_qmt_raw = img_qmt_raw.get_fdata()
else:
    logger.info(f'Stacking raw images into single 4D')
    data_qmt_raw = np.concat((data_mt0_raw, data_mtw_raw, data_pdw_raw, data_t1w_raw), axis=3)
    logger.info(f'data_qmt_raw shape is {data_qmt_raw.shape}')

    img_qmt_raw = nib.Nifti1Image(
        dataobj=data_qmt_raw,
        affine=img_t1w_raw[0].affine,
        header=img_t1w_raw[0].header)
    img_qmt_raw.header['dim'][0] = 4
    logger.info(f'Writing raw 4D images in disk: {path_qmt_raw}'); nib.save(img_qmt_raw, path_qmt_raw)


#%% Run tMPPCA

path_qmt_den = denoising_dir / '4D_den.nii'
if path_qmt_den.exists():
    logger.info(f'tMPPCA already done : {path_qmt_den}')
else:
    cmd = f'denoise-tmppca \
            --num_threads {os.cpu_count()} \
            --window 7,7,7 \
            {path_qmt_raw} \
            {path_qmt_den} '
    subprocess.run(cmd, shell=True)

img_qmt_den = nib.load(path_qmt_den)
data_qmt_den = img_qmt_den.get_fdata()


#%% De-concatenante

n = n_mt0 + n_mtw + n_pdw + n_t1w

for idx in range(n):
    if raw_qmt[idx].den_nii.exists():
        logger.info(f'Already exist: {raw_qmt[idx].den_nii.stem}[.nii, .json]')
    else:
        data= data_qmt_den[:,:,:,idx]
        img_den = nib.Nifti1Image(
            dataobj=data,
            affine=img_qmt_den.affine,
            header=img_qmt_den.header)
        logger.info(f'Writing {raw_qmt[idx].den_nii.stem}[.nii, .json]')
        nib.save(img_den, raw_qmt[idx].den_nii)
        shutil.copyfile(raw_qmt[idx].raw_json,raw_qmt[idx].den_json)


#%% Masking

for con in Contrast:
    fpath = denoising_dir / f'den_{con}_e1.nii'
    mask  = masking_dir   / f'mask_{fpath.name}'
    if mask.exists():
        logger.info(f'Already exist: {mask}')
    else:
        subprocess.run(f'mri_synthstrip --image {fpath} --mask {mask} --border 2', shell=True)

#%% Motion correction
        
target_img = denoising_dir /      'den_t1w_e1.nii'
target_msk = masking_dir   / 'mask_den_t1w_e1.nii'

for con in Contrast:
    if con is Contrast.t1w: continue
    
    moving_img = denoising_dir / f'den_{con}_e1.nii'
    moving_msk = masking_dir   / f'mask_{fpath.name}'

    mat = moco_dir / f'ants_{con}_0GenericAffine.mat'
    if not mat.exists():
        logger.info(f'antsRegistration {con} -> t1w : {moving_img}')

        cmd = f'antsRegistration \
                --dimensionality 3 \
                --initial-moving-transform [{target_msk},{moving_msk},1] \
                --winsorize-image-intensities [0.005,0.995] \
                --use-histogram-matching 0 \
                --masks [{target_msk}, {moving_msk}] \
                --output {moco_dir}/ants_{con}_ \
                --collapse-output-transforms 1 \
                --interpolation LanczosWindowedSinc \
                --transform Rigid[0.1] \
                --metric MI[{target_img},{moving_img},1,32,Regular,0.25] \
                --convergence [500x200,1e-6,10] \
                --shrink-factors   2x1 \
                --smoothing-sigmas 1x0vox'

        subprocess.run(cmd, shell=True)

    moco_img = moco_dir / f'moco_den_{con}_e1.nii'
    if not moco_img.exists():
        logger.info(f'antsApplyTransforms {con} -> t1w : {moco_img}')

        cmd = f'antsApplyTransforms \
                --dimensionality 3 \
                --interpolation LanczosWindowedSinc \
                --input {moving_img}\
                --reference-image {moving_img} \
                --transform {mat} \
                --output {moco_img}'

        subprocess.run(cmd, shell=True)


#%% B1map

raw_b1map_workdir = b1map_dir / 'b1map_0_raw.nii'
b1map_resliced    = b1map_dir / 'b1map_1_resliced.nii'
b1map_scaled      = b1map_dir / 'b1map_2_scaled.nii'
b1map_smoothed    = b1map_dir / 'b1map_3_smoothed.nii'
if b1map_smoothed.exists():
    logger.info(f'B1map ready: {b1map_smoothed}')
else:
    logger.info(f'Preparing B1map : reslice, scale, smooth')
    shutil.copyfile(raw_b1map.raw_nii,raw_b1map_workdir)
    cmd = f'antsApplyTransforms \
            --dimensionality 3 \
            --interpolation LanczosWindowedSinc \
            --input {raw_b1map_workdir}\
            --reference-image {target_img} \
            --output {b1map_resliced}'
    subprocess.run(cmd, shell=True)
    subprocess.run(f'ImageMath 3 {b1map_scaled} m {b1map_resliced} {b1map_factor}', shell=True)
    subprocess.run(f'SmoothImage 3 {b1map_scaled} 8 {b1map_smoothed}', shell=True)


#%% Fit

mt0 = moco_dir      / 'moco_den_mt0_e1.nii'
mtw = moco_dir      / 'moco_den_mtw_e1.nii'
pdw = moco_dir      / 'moco_den_pdw_e1.nii'
t1w = denoising_dir /      'den_t1w_e1.nii'
mpf = results_dir   / 'mpf.nii'
t1f = results_dir   / 't1f.nii'
r1f = results_dir   / 'r1f.nii'

if t1f.exists():
    logger.info(f'fit-JSPqMT done: {t1f}')
else:
    cmd = f'fit-JSPqMT {mt0},{mtw} {pdw},{t1w} {mpf} {t1f} \
            --R1f {r1f} \
            --mask {target_msk} \
            --B1 {b1map_smoothed} \
            --MTw_TIMINGS 12.0,2.1,0.25,30.0 \
            --VFA_TIMINGS 0.25,30.0 \
            --VFA_PARX 6.0,33.0,BP \
            --MTw_PARX 14.0,BP,560.0,4000.0,Hann-Sine \
            --qMTconstraint_PARX 0.0158,10.0e-6,21.1 \
            --use_GBM \
            --nworkers {os.cpu_count()} \
            --cpp_opt'
    subprocess.run(cmd, shell=True)

