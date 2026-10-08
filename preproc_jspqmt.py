#%% Imports

import nibabel as nib
import numpy as np

import logging, os, re, json, sys, enum, pathlib, shutil, subprocess


#%% Config

main_dir = pathlib.Path('/network/iss/cenir/analyse/irm/users/benoit.beranger/2026_07_30_DEV2_444_09_compareMPM')

path_size_pca = 5 # default value 5, while 7 is noisier
rms_num_echos = 1 # use first echo, no need for RMS with 3 echos

work_dir = main_dir / f'jspqmt_c9_patch{path_size_pca}_rms{rms_num_echos}'
regex_mt0 = re.compile(r'.*vibeMT_mt0_sag_6eco_1iso_c9$')
regex_mtw = re.compile(r'.*vibeMT_mtw_sag_3eco_1iso_c9$')
regex_pdw = re.compile(r'.*vibeMT_pdw_sag_6eco_1iso_c9$')
regex_t1w = re.compile(r'.*vibeMT_t1w_sag_6eco_1iso_c9$')

# work_dir = main_dir / f'jspqmt_c9_es_patch{path_size_pca}_rms{rms_num_echos}'
# regex_mt0 = re.compile(r'.*vibeMT_mt0_sag_6eco_1iso_c9_es$')
# regex_mtw = re.compile(r'.*vibeMT_mtw_sag_3eco_1iso_c9_es$')
# regex_pdw = re.compile(r'.*vibeMT_pdw_sag_6eco_1iso_c9_es$')
# regex_t1w = re.compile(r'.*vibeMT_t1w_sag_6eco_1iso_c9_es$')

# work_dir = main_dir / f'jspqmt_c6_patch{path_size_pca}_rms{rms_num_echos}'
# regex_mt0 = re.compile(r'.*vibeMT_mt0_sag_6eco_1iso_c6$')
# regex_mtw = re.compile(r'.*vibeMT_mtw_sag_3eco_1iso_c6$')
# regex_pdw = re.compile(r'.*vibeMT_pdw_sag_6eco_1iso_c6$')
# regex_t1w = re.compile(r'.*vibeMT_t1w_sag_6eco_1iso_c6$')

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
    def main_dir (self): return self._main_dir
    @property
    def root     (self): return self._root
    @property
    def raw_nii  (self): return self._raw_nii
    @property
    def raw_json (self): return self._raw_json
    @property
    def den_nii  (self): return self._den_nii
    @property
    def den_json (self): return self._den_json
    @property
    def moco_nii (self): return self._moco_nii
    @property
    def moco_json(self): return self._moco_json
    
    @main_dir.setter
    def main_dir (self, value): self._main_dir  = pathlib.Path(value)
    @root.setter
    def root     (self, value): self._root      = pathlib.Path(value)
    @raw_nii.setter
    def raw_nii  (self, value): self._raw_nii   = pathlib.Path(value)
    @raw_json.setter
    def raw_json (self, value): self._raw_json  = pathlib.Path(value)
    @den_nii.setter
    def den_nii  (self, value): self._den_nii   = pathlib.Path(value)
    @den_json.setter
    def den_json (self, value): self._den_json  = pathlib.Path(value)
    @moco_nii.setter
    def moco_nii (self, value): self._moco_nii  = pathlib.Path(value)
    @moco_json.setter
    def moco_json(self, value): self._moco_json = pathlib.Path(value)

class Contrast(enum.StrEnum):
    mt0 = 'mt0'
    mtw = 'mtw'
    pdw = 'pdw'
    t1w = 't1w'


#%% Prepare dirs

work_dir.mkdir(parents=True, exist_ok=True)
logger.info(f'work_dir is : {work_dir}')

denoising_dir = work_dir / '01_Denoising'
rms_dir       = work_dir / '02_RMS'
masking_dir   = work_dir / '03_Masking'
moco_dir      = work_dir / '04_MotionCorrection'
b1map_dir     = work_dir / '05_B1map'
r2s_dir       = work_dir / '05_R2s'
denoising_dir.mkdir(parents=True, exist_ok=True)
rms_dir      .mkdir(parents=True, exist_ok=True)
masking_dir  .mkdir(parents=True, exist_ok=True)
moco_dir     .mkdir(parents=True, exist_ok=True)
b1map_dir    .mkdir(parents=True, exist_ok=True)
r2s_dir      .mkdir(parents=True, exist_ok=True)

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
                new.moco_nii = moco_dir / f'moco_{new.den_nii.name}'
                new.moco_json = new.moco_nii.with_suffix('.json')
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
if n_mt0 < rms_num_echos: raise ValueError(f'not enough {n_mt0}/{rms_num_echos} echos for mt0')
if n_mtw < rms_num_echos: raise ValueError(f'not enough {n_mt0}/{rms_num_echos} echos for mtw')
if n_pdw < rms_num_echos: raise ValueError(f'not enough {n_mt0}/{rms_num_echos} echos for pdw')
if n_t1w < rms_num_echos: raise ValueError(f'not enough {n_mt0}/{rms_num_echos} echos for t1w')

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
            --window {path_size_pca},{path_size_pca},{path_size_pca} \
            {path_qmt_raw} \
            {path_qmt_den} '
    subprocess.run(cmd, shell=True)

img_qmt_den = nib.load(path_qmt_den)
data_qmt_den = img_qmt_den.get_fdata()


#%% De-concatenante

for idx in range(len(raw_qmt)):
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


#%% Root Mean Square of the first echos

for con in Contrast:
    fpath = rms_dir / f'den_{con}_rms.nii'
    if fpath.exists():
        logger.info(f'Already exist: {fpath}')
    else:
        file_stack = list(filter(lambda x: x.contrast==con, raw_mag))
        file_stack = file_stack[:rms_num_echos]
        img_stack = [nib.load(file.root / file.den_nii) for file in file_stack]
        data_stack = np.stack([img.get_fdata() for img in img_stack], axis=3)
        rms = np.sqrt(np.mean(np.square(data_stack),axis=3))
        img_rms = nib.Nifti1Image(
            dataobj=rms,
            affine=img_stack[0].affine,
            header=img_stack[0].header)
        logger.info(f'Writing {fpath}')
        nib.save(img_rms, fpath)


#%% Masking

for con in Contrast:
    fpath = rms_dir / f'den_{con}_rms.nii'
    mask  = masking_dir   / f'mask_{fpath.name}'
    if mask.exists():
        logger.info(f'Already exist: {mask}')
    else:
        subprocess.run(f'mri_synthstrip --image {fpath} --mask {mask} --border 2', shell=True)


#%% Motion correction
        
target_img = rms_dir     /      'den_t1w_rms.nii'
target_msk = masking_dir / 'mask_den_t1w_rms.nii'

for con in Contrast:
    if con is Contrast.t1w: continue
    
    moving_img = rms_dir     / f'den_{con}_rms.nii'
    moving_msk = masking_dir / f'mask_{fpath.name}'

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

    moco_img = moco_dir / f'moco_den_{con}_rms.nii'
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

    if   con is Contrast.pdw:  n_echo = n_pdw
    elif con is Contrast.t1w:  n_echo = n_t1w
    elif con is Contrast.mtw:  n_echo = n_mtw
    elif con is Contrast.mt0:  n_echo = n_mt0
    else: raise RuntimeError('contrast ?')

    for echo in range(1,n_echo+1):
        file_in  = denoising_dir /      f'den_{con}_e{echo}.nii'
        file_out = moco_dir      / f'moco_den_{con}_e{echo}.nii'
        if not file_out.exists():
            logger.info(f'antsApplyTransforms {con} -> t1w : {file_out}')
            cmd = f'antsApplyTransforms \
                    --dimensionality 3 \
                    --interpolation LanczosWindowedSinc \
                    --input {file_in}\
                    --reference-image {file_in} \
                    --transform {mat} \
                    --output {file_out}'
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

mt0 = moco_dir    / 'moco_den_mt0_rms.nii'
mtw = moco_dir    / 'moco_den_mtw_rms.nii'
pdw = moco_dir    / 'moco_den_pdw_rms.nii'
t1w = rms_dir     /      'den_t1w_rms.nii'
mpf = results_dir / 'mpf.nii'
t1f = results_dir / 't1f.nii'
r1f = results_dir / 'r1f.nii'

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


#%% fit R2* // functions

def fit_t2s_loglin_leastsquare(data: np.ndarray,
                               TE  : np.ndarray,
                               mask: np.ndarray,
                               method: str
                               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] :

    nx,ny,nz,ne = data.shape
    nvox = nx*ny*nz
    
    data = np.permute_dims(data, (3,0,1,2)) # echos on the first dimension, easier reshape

    data_2d = np.reshape(data, (ne,nvox))
    mask_flat = np.reshape(mask, (nvox))
    
    mask_flat = mask_flat > 0.1
    data_masked = data_2d[:,mask_flat]

    # AX=B : here, X is [log(S0) -R2s]
    b = np.log(data_masked) # [ne, nvox]
    b[~np.isfinite(b)] = 0
    a = np.vstack( (np.ones(ne), TE) ).T # [ne, 2]

    if method == 'ols':
        beta, err_flat, _, _ = np.linalg.lstsq(a, b)
    elif method == 'wls':
        # weighted least square : use 1/TE as weight, as later echos have less SNR
        w = np.diag(np.sqrt(1/TE))
        wa = np.dot(w,a)
        wb = np.dot(w,b) 
        beta, err_flat, _, _ = np.linalg.lstsq(wa, wb)
    else:
        raise RuntimeError(f'bad t2s fit method: {method}')

    s0_flat = np.exp(beta[0,:])

    r2s_flat = -beta[1,:]
    t2s_flat = 1/r2s_flat
    r2s_flat[r2s_flat<0] = 0
    t2s_flat[t2s_flat>1] = 0
    t2s_flat[t2s_flat<0] = 0

    s0  = np.zeros(nvox, dtype=np.float32)
    r2s = np.zeros(nvox, dtype=np.float32)
    t2s = np.zeros(nvox, dtype=np.float32)
    err = np.zeros(nvox, dtype=np.float32)
    s0 [mask_flat] = s0_flat
    r2s[mask_flat] = r2s_flat
    t2s[mask_flat] = t2s_flat
    err[mask_flat] = err_flat
    s0  =  s0.reshape(nx,ny,nz)
    r2s = r2s.reshape(nx,ny,nz)
    t2s = t2s.reshape(nx,ny,nz)
    err = err.reshape(nx,ny,nz)

    return s0, t2s, r2s, err


def fit_t2s_estatics(data_mt0: np.ndarray, data_mtw: np.ndarray, data_pdw: np.ndarray, data_t1w: np.ndarray,
                       TE_mt0: np.ndarray,   TE_mtw: np.ndarray,   TE_pdw: np.ndarray,   TE_t1w: np.ndarray,
                     mask: np.ndarray,
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] :

    N_CONTRAST = 4

    nx,ny,nz = data_mt0.shape[:3]
    nvox = nx*ny*nz
    
    mask_flat = np.reshape(mask, (nvox))
    mask_flat = mask_flat > 0.1

    data_mt0 = np.permute_dims(data_mt0, (3,0,1,2))
    data_mtw = np.permute_dims(data_mtw, (3,0,1,2))
    data_pdw = np.permute_dims(data_pdw, (3,0,1,2))
    data_t1w = np.permute_dims(data_t1w, (3,0,1,2))

    ne_mt0 = len(TE_mt0)
    ne_mtw = len(TE_mtw)
    ne_pdw = len(TE_pdw)
    ne_t1w = len(TE_t1w)

    data_2d_mt0 = np.reshape(data_mt0, (ne_mt0,nvox))
    data_2d_mtw = np.reshape(data_mtw, (ne_mtw,nvox))
    data_2d_pdw = np.reshape(data_pdw, (ne_pdw,nvox))
    data_2d_t1w = np.reshape(data_t1w, (ne_t1w,nvox))
    
    data_masked_mt0 = data_2d_mt0[:,mask_flat]
    data_masked_mtw = data_2d_mtw[:,mask_flat]
    data_masked_pdw = data_2d_pdw[:,mask_flat]
    data_masked_t1w = data_2d_t1w[:,mask_flat]

    a_mt0 = np.zeros((ne_mt0, 1+N_CONTRAST))
    a_mtw = np.zeros((ne_mtw, 1+N_CONTRAST))
    a_pdw = np.zeros((ne_pdw, 1+N_CONTRAST))
    a_t1w = np.zeros((ne_t1w, 1+N_CONTRAST))

    a_mt0[:,0] = 1
    a_mtw[:,1] = 1
    a_pdw[:,2] = 1
    a_t1w[:,3] = 1

    a_mt0[:,-1] = TE_mt0
    a_mtw[:,-1] = TE_mtw
    a_pdw[:,-1] = TE_pdw
    a_t1w[:,-1] = TE_t1w

    a = np.vstack((          a_mt0,           a_mtw,           a_pdw,           a_t1w))
    b = np.vstack((data_masked_mt0, data_masked_mtw, data_masked_pdw, data_masked_t1w))
    b = np.log(b)
    b[~np.isfinite(b)] = 0

    beta, err_flat, _, _ = np.linalg.lstsq( a, b )

    s0_flat_mt0 = np.exp(beta[0,:])
    s0_flat_mtw = np.exp(beta[1,:])
    s0_flat_pdw = np.exp(beta[2,:])
    s0_flat_t1w = np.exp(beta[3,:])

    r2s_flat = -beta[-1,:]
    t2s_flat = 1/r2s_flat
    r2s_flat[r2s_flat<0] = 0
    t2s_flat[t2s_flat>1] = 0
    t2s_flat[t2s_flat<0] = 0

    s0_mt0  = np.zeros(nvox, dtype=np.float32)
    s0_mtw  = np.zeros(nvox, dtype=np.float32)
    s0_pdw  = np.zeros(nvox, dtype=np.float32)
    s0_t1w  = np.zeros(nvox, dtype=np.float32)
    r2s     = np.zeros(nvox, dtype=np.float32)
    t2s     = np.zeros(nvox, dtype=np.float32)
    err     = np.zeros(nvox, dtype=np.float32)
    s0_mt0[mask_flat] = s0_flat_mt0
    s0_mtw[mask_flat] = s0_flat_mtw
    s0_pdw[mask_flat] = s0_flat_pdw
    s0_t1w[mask_flat] = s0_flat_t1w
    r2s   [mask_flat] = r2s_flat
    t2s   [mask_flat] = t2s_flat
    err   [mask_flat] = err_flat
    s0_mt0  = s0_mt0.reshape(nx,ny,nz)
    s0_mtw  = s0_mtw.reshape(nx,ny,nz)
    s0_pdw  = s0_pdw.reshape(nx,ny,nz)
    s0_t1w  = s0_t1w.reshape(nx,ny,nz)
    r2s     = r2s   .reshape(nx,ny,nz)
    t2s     = t2s   .reshape(nx,ny,nz)
    err     = err   .reshape(nx,ny,nz)

    return s0_mt0, s0_mtw, s0_pdw, s0_t1w, t2s, r2s, err


#%% fit R2* // ESTATICS

r2s_estatics = results_dir / f'r2s_estatics.nii'
if r2s_estatics.exists():
    logger.info(f'Already exist: {r2s_estatics}')
else:
    img_mt0 = [nib.load(file.root / file.moco_nii) for file in raw_mt0]
    img_mtw = [nib.load(file.root / file.moco_nii) for file in raw_mtw]
    img_pdw = [nib.load(file.root / file.moco_nii) for file in raw_pdw]
    img_t1w = [nib.load(file.root / file. den_nii) for file in raw_t1w]
    data_mt0 = np.stack([img.get_fdata() for img in img_mt0], axis=3)
    data_mtw = np.stack([img.get_fdata() for img in img_mtw], axis=3)
    data_pdw = np.stack([img.get_fdata() for img in img_pdw], axis=3)
    data_t1w = np.stack([img.get_fdata() for img in img_t1w], axis=3)
    mask = nib.load(target_msk).get_fdata()
    te_mt0 = np.array([file.echo_time for file in raw_mt0])
    te_mtw = np.array([file.echo_time for file in raw_mtw])
    te_pdw = np.array([file.echo_time for file in raw_pdw])
    te_t1w = np.array([file.echo_time for file in raw_t1w])
    
    s0_mt0, s0_mtw, s0_pdw, s0_t1w, t2s, r2s, err = fit_t2s_estatics(
        data_mt0=data_mt0, data_mtw=data_mtw, data_pdw=data_pdw, data_t1w=data_t1w,
          TE_mt0=  te_mt0,   TE_mtw=  te_mtw,   TE_pdw=  te_pdw,   TE_t1w=  te_t1w,
        mask=mask
        )
    
    nib.save(nib.Nifti1Image(dataobj=r2s   , affine=img_mt0[0].affine, header=img_mt0[0].header, dtype=np.float32), results_dir / f'r2s_estatics.nii')
    nib.save(nib.Nifti1Image(dataobj=t2s   , affine=img_mt0[0].affine, header=img_mt0[0].header, dtype=np.float32), results_dir / f't2s_estatics.nii')
    nib.save(nib.Nifti1Image(dataobj=s0_mt0, affine=img_mt0[0].affine, header=img_mt0[0].header, dtype=np.float32), r2s_dir     / f's0_mt0_estatics.nii')
    nib.save(nib.Nifti1Image(dataobj=s0_mtw, affine=img_mt0[0].affine, header=img_mt0[0].header, dtype=np.float32), r2s_dir     / f's0_mtw_estatics.nii')
    nib.save(nib.Nifti1Image(dataobj=s0_pdw, affine=img_mt0[0].affine, header=img_mt0[0].header, dtype=np.float32), r2s_dir     / f's0_pdw_estatics.nii')
    nib.save(nib.Nifti1Image(dataobj=s0_t1w, affine=img_mt0[0].affine, header=img_mt0[0].header, dtype=np.float32), r2s_dir     / f's0_t1w_estatics.nii')


#%% fit R2* // OLS + WLS + NumART on all contrasts, just to compare

for con in Contrast:

    r2s_ols = r2s_dir / f'r2s_{con}_ols.nii'
    if r2s_ols.exists():
        logger.info(f'Already exist: {r2s_ols}')
    else:
        if   con is Contrast.mt0:  files = raw_mt0
        elif con is Contrast.mtw:  files = raw_mtw
        elif con is Contrast.pdw:  files = raw_pdw
        elif con is Contrast.t1w:  files = raw_t1w
        imgs = [nib.load(file.root / file. den_nii) for file in files]
        data = np.stack([img.get_fdata() for img in imgs], axis=3)
        mask = nib.load(target_msk).get_fdata()
        te = np.array([file.echo_time for file in files])

        s0, t2s, r2s, err = fit_t2s_loglin_leastsquare(data, te, mask, 'ols')
        nib.save(nib.Nifti1Image(dataobj=r2s, affine=imgs[0].affine, header=imgs[0].header, dtype=np.float32), r2s_dir / f'r2s_{con}_ols.nii')
        nib.save(nib.Nifti1Image(dataobj=t2s, affine=imgs[0].affine, header=imgs[0].header, dtype=np.float32), r2s_dir / f't2s_{con}_ols.nii')
        nib.save(nib.Nifti1Image(dataobj=s0 , affine=imgs[0].affine, header=imgs[0].header, dtype=np.float32), r2s_dir /  f's0_{con}_ols.nii')

        s0, t2s, r2s, err = fit_t2s_loglin_leastsquare(data, te, mask, 'wls')
        nib.save(nib.Nifti1Image(dataobj=r2s, affine=imgs[0].affine, header=imgs[0].header, dtype=np.float32), r2s_dir / f'r2s_{con}_wls.nii')
        nib.save(nib.Nifti1Image(dataobj=t2s, affine=imgs[0].affine, header=imgs[0].header, dtype=np.float32), r2s_dir / f't2s_{con}_wls.nii')
        nib.save(nib.Nifti1Image(dataobj=s0 , affine=imgs[0].affine, header=imgs[0].header, dtype=np.float32), r2s_dir /  f's0_{con}_wls.nii')

        t2s = (te[-1]-te[0]) / (2*(len(te)-1)) \
        * ( (data[:,:,:,0] + data[:,:,:,-1]) + 2 * np.sum(data[:,:,:,1:-1], axis=3) ) \
        /   (data[:,:,:,0] - data[:,:,:,-1])
        t2s *= mask
        r2s = 1/t2s
        t2s[t2s<0]    = 0
        t2s[t2s>1]    = 0
        r2s[r2s<0]    = 0
        r2s[r2s>1000] = 0
        nib.save(nib.Nifti1Image(dataobj=r2s, affine=imgs[0].affine, header=imgs[0].header, dtype=np.float32), r2s_dir / f'r2s_{con}_numart.nii')
        nib.save(nib.Nifti1Image(dataobj=t2s, affine=imgs[0].affine, header=imgs[0].header, dtype=np.float32), r2s_dir / f't2s_{con}_numart.nii')
