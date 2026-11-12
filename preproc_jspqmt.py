#%% Imports

import nibabel as nib
import numpy as np

import logging, os, re, json, sys, enum, pathlib, shutil, subprocess


#%% Config

main_dir = pathlib.Path('/network/iss/cenir/analyse/irm/users/benoit.beranger/2026_07_30_DEV2_444_09_compareMPM')

path_size_pca = 5 # default value 5, while 7 is noisier
rms_num_echos = 1 # use first echo, no need for RMS with 3 echos

JSPqMT: dict = {
    'MTw_TIMINGS':{
        'SatPulseDuration'     :      12.0  , # Saturation pulse duration (ms)
        'Interdelay'           :       2.1  , # Interdelay between Saturation pulse and Readout pulse (ms)
        'ReadoutPulseDuration' :       0.250, # Readout pulse duration (ms)
        'TR'                   :      30.0  , # Sequence Time-to-Repetition (TR; ms)
    },
    'VFA_TIMINGS':{
        'ReadoutPulseDuration' :       0.250, # Readout pulse duration (ms)
        'TR'                   :      30.0  , # Sequence Time-to-Repetition (TR; ms)
    },
    'MTw_PARX':{
        'ReadoutFA'            :      14.0  , # Readout flip angle of MT0/MTw (deg; single common value)
        'ReadoutPulseShape'    : 'BP'       , # Readout pulse shape (Hann, BP)
        'SaturationFA'         :     560.0  , # Saturation pulse flip angle (deg)
        'SaturationOffset'     :    4000.0  , # Saturation pulse off-resonance frequency (Hz)
        'SaturationPulseShape' : 'Hann-Sine', # Saturation pulse shape (Hann-Sine, GaussHann-Sine, Gauss-Sine)
    },
    'VFA_PARX':{
        'ReadoutFA'            : [6.0, 33.0], # Readout flip angles [VFA1, VFA2, ..., VFAn] (deg; same order as in provided VFA volume(s))
        'ReadoutPulseShape'    : 'BP'       , # Readout pulse shape (Hann, BP)
    }
}

work_dir = main_dir / f'jspqmt_c9_patch{path_size_pca}_rms{rms_num_echos}_unwrap_cplx'
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


class Contrast(enum.StrEnum):
    mt0 = 'mt0'
    mtw = 'mtw'
    pdw = 'pdw'
    t1w = 't1w'


class File:

    def __init__(self) -> None:
        self.is_mag         : bool     = None
        self.is_b1map       : bool     = False
        self.contrast       : Contrast = None
        self.echo_number    : int      = None
        self.echo_time      : float    = None
        self.info           : dict     = None

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


class ContrastMP:
    def __init__(self):
        self.m: list = []
        self.p: list = []

    def __len__(self):
        return len(self.m)


class MPMContainer(dict):
    def __init__(self):
        self.mt0: ContrastMP = ContrastMP()
        self.mtw: ContrastMP = ContrastMP()
        self.pdw: ContrastMP = ContrastMP()
        self.t1w: ContrastMP = ContrastMP()

        self._count : int = -1
        self._offset: int =  0

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __len__(self):
        return len(self.mt0) + len(self.mtw) + len(self.pdw) + len(self.t1w)


#%% Prepare dirs

work_dir.mkdir(parents=True, exist_ok=True)
logger.info(f'work_dir is : {work_dir}')

unwrap_dir    = work_dir / '01_Unwrap'
denoising_dir = work_dir / '02_Denoising'
rms_dir       = work_dir / '03_RMS'
masking_dir   = work_dir / '04_Masking'
moco_dir      = work_dir / '05_MotionCorrection'
b1map_dir     = work_dir / '06_B1map'
r2s_dir       = work_dir / '07_R2s'
qsm_dir       = work_dir / '08_QSM'
unwrap_dir   .mkdir(parents=True, exist_ok=True)
denoising_dir.mkdir(parents=True, exist_ok=True)
rms_dir      .mkdir(parents=True, exist_ok=True)
masking_dir  .mkdir(parents=True, exist_ok=True)
moco_dir     .mkdir(parents=True, exist_ok=True)
b1map_dir    .mkdir(parents=True, exist_ok=True)
r2s_dir      .mkdir(parents=True, exist_ok=True)
qsm_dir      .mkdir(parents=True, exist_ok=True)

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
                if   new.info['ImageType'][2] == 'M': new.is_mag = True;  mp = 'm'
                elif new.info['ImageType'][2] == 'P': new.is_mag = False; mp = 'p' 
                new.echo_number = new.info['EchoNumber']
                new.echo_time   = new.info['EchoTime']
                if regex_mt0.match(root): new.contrast = Contrast.mt0
                if regex_mtw.match(root): new.contrast = Contrast.mtw
                if regex_pdw.match(root): new.contrast = Contrast.pdw
                if regex_t1w.match(root): new.contrast = Contrast.t1w
                new.den_nii = denoising_dir /  f'den_{new.contrast}_e{new.echo_number}_{mp}.nii'
                new.den_json = new.den_nii.with_suffix('.json')
                new.moco_nii = moco_dir / f'moco_{new.den_nii.name}'
                new.moco_json = new.moco_nii.with_suffix('.json')
                raw_sources.append(new)
            if regex_b1map.match(root):
                new.is_b1map = True
                raw_sources.append(new)

raw_mag   = list(filter(lambda x: x.is_mag==True , raw_sources))
raw_pha   = list(filter(lambda x: x.is_mag==False, raw_sources))
files = MPMContainer()
for con in Contrast:
    files[con].m = list(filter(lambda x: x.contrast==con, raw_mag))
    files[con].p = list(filter(lambda x: x.contrast==con, raw_pha))
    files[con].m = sorted(files[con].m)
    files[con].p = sorted(files[con].p)
    if len(files[con].m) < 1: raise FileNotFoundError(f'{con} // magnitude not found with {main_dir / regex_mt0.pattern / regex_rawnifti.pattern}')
    if len(files[con].p) < 1: raise FileNotFoundError(f'{con} // phase     not found with {main_dir / regex_mt0.pattern / regex_rawnifti.pattern}')
    logger.info(f'{con} // found {len(files[con].m)} magnitudes')
    logger.info(f'{con} // found {len(files[con].p)} phases')
    if len(files[con].m) < rms_num_echos: raise ValueError(f'{con} // not enough echos for RMS: {len(files[con].m)}/{rms_num_echos}')
    logger.info(f'{con} // sorted TE = {[round(file.echo_time*1000,3) for file in files[con].m]}(ms)')

raw_b1map: list[File] = list(filter(lambda x: x.is_b1map==True, raw_sources))
if len(raw_b1map) < 1: raise FileNotFoundError(f'no raw_b1map found with {main_dir / regex_b1map.pattern / regex_rawnifti.pattern}')
if len(raw_b1map) > 1: raise FileNotFoundError(f'too many raw_b1map found with {main_dir / regex_b1map.pattern / regex_rawnifti.pattern}')
raw_b1map: File = raw_b1map[0]


#%% Prepare some containers and shortcuts

fileorder_4D_m:list[File] = []
fileorder_4D_p:list[File] = []
for con in Contrast:
    for file in files[con].m: fileorder_4D_m.append(file)
    for file in files[con].p: fileorder_4D_p.append(file) 


#%% Unwrap the phase
for con in Contrast:

    mag  = files[con].m[0].raw_nii
    mask = unwrap_dir / f'mask_{con}.nii'
    if mask.exists():
        logger.info(f'Already exist: {mask}')
    else:
        subprocess.run(f'mri_synthstrip --image {mag} --mask {mask} --border 2', shell=True)

    for file in files[con].p:
        phase_in = file.raw_nii
        phase_out = unwrap_dir / f'unwrap_{con}_e{file.echo_number}_p.nii'
        if not phase_out.exists():
            cmd = f'qsmxt unwrap laplacian \
                    --mask {mask}          \
                    --output {phase_out}   \
                    {phase_in}'
            subprocess.run(cmd, shell=True)


#%% Concatenate all raw images (mag+pĥa)

path_4D_raw = denoising_dir / '4D_raw.nii'
if path_4D_raw.exists():
    logger.info(f'Already done raw 4D : {path_4D_raw}')    
else:
    logger.info(f'Stacking raw images into single 4D')
    
    img_raw    = MPMContainer()
    data_raw   = MPMContainer()
    data_raw_c = MPMContainer()

    for con in Contrast:
        logger.info(f'Loading header files[{con}].m');  img_raw[con].m = [nib.load(file.root / file.raw_nii) for file in   files[con].m]
        logger.info(f'Loading header files[{con}].p');  img_raw[con].p = [nib.load(unwrap_dir / f'unwrap_{con}_e{file.echo_number}_p.nii') for file in   files[con].p]
        logger.info(f'Loading data   files[{con}].m'); data_raw[con].m = np.stack([img.get_fdata()           for img  in img_raw[con].m], axis=3)
        logger.info(f'Loading data   files[{con}].p'); data_raw[con].p = np.stack([img.get_fdata()           for img  in img_raw[con].p], axis=3)
        # data_raw[con].p *= np.pi / 4096 # rescale Siemens phase [-4096 4095] to radian
        data_raw_c[con] = data_raw[con].m * np.exp(1j * data_raw[con].p)

    data_raw_ri = np.concat((np.real(data_raw_c.mt0), np.real(data_raw_c.mtw), np.real(data_raw_c.pdw), np.real(data_raw_c.t1w),
                             np.imag(data_raw_c.mt0), np.imag(data_raw_c.mtw), np.imag(data_raw_c.pdw), np.imag(data_raw_c.t1w)), axis=3)
    img_raw_ri = nib.Nifti1Image(
            dataobj=data_raw_ri,
            affine=img_raw.mt0.m[0].affine,
            header=img_raw.mt0.m[0].header,
            dtype=np.float32)
    img_raw_ri.header['dim'][0] = 4
    logger.info(f'Writing raw Re Im 4D images in disk: {path_4D_raw}');
    nib.save(img_raw_ri, path_4D_raw)


#%% Run tMPPCA

path_4D_den = denoising_dir / '4D_den.nii'
if path_4D_den.exists():
    logger.info(f'tMPPCA already done : {path_4D_den}')
else:
    cmd = f'denoise-tmppca \
            --mask {mask} \
            --num_threads {os.cpu_count()} \
            --window {path_size_pca},{path_size_pca},{path_size_pca} \
            {path_4D_raw} \
            {path_4D_den} '
    subprocess.run(cmd, shell=True)


#%% De-concatenante

for idx in range(len(files)):
    if fileorder_4D_m[idx].den_nii.exists():
        logger.info(f'Already exist: {fileorder_4D_m[idx].den_nii.stem}[.nii, .json]')
    else:
        img_den_ri = nib.load(path_4D_den)
        data_den_ri = img_den_ri.get_fdata()
        data_r = data_den_ri[:,:,:,idx             ]
        data_i = data_den_ri[:,:,:,idx + len(files)]
        data_c = data_r + 1j*data_i
        refimg = nib.load(fileorder_4D_m[idx].raw_nii)
        img_den_m = nib.Nifti1Image(dataobj=np.abs  (data_c), affine=refimg.affine, header=refimg.header)
        img_den_p = nib.Nifti1Image(dataobj=np.angle(data_c), affine=refimg.affine, header=refimg.header)
        logger.info(f'Writing MAG {fileorder_4D_m[idx].den_nii.stem}[.nii, .json]'); nib.save(img_den_m, fileorder_4D_m[idx].den_nii); shutil.copyfile(fileorder_4D_m[idx].raw_json,fileorder_4D_m[idx].den_json)
        logger.info(f'Writing PHA {fileorder_4D_p[idx].den_nii.stem}[.nii, .json]'); nib.save(img_den_p, fileorder_4D_p[idx].den_nii); shutil.copyfile(fileorder_4D_p[idx].raw_json,fileorder_4D_p[idx].den_json)


#%% Root Mean Square of the first echos

for con in Contrast:
    fpath = rms_dir / f'den_{con}_rms.nii'
    if fpath.exists():
        logger.info(f'Already exist: {fpath}')
    else:
        file_stack = files[con].m
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
        
target_img = rms_dir     /      'den_mt0_rms.nii'
target_msk = masking_dir / 'mask_den_mt0_rms.nii'

for con in Contrast:
    if con is Contrast.mt0: continue
    
    moving_img = rms_dir     / f'den_{con}_rms.nii'
    moving_msk = masking_dir / f'mask_{fpath.name}'

    mat = moco_dir / f'ants_{con}_0GenericAffine.mat'
    if not mat.exists():
        logger.info(f'antsRegistration {con} -> mt0 : {moving_img}')

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
        logger.info(f'antsApplyTransforms {con} -> mt0 : {moco_img}')
        cmd = f'antsApplyTransforms \
                --dimensionality 3 \
                --interpolation LanczosWindowedSinc \
                --input {moving_img}\
                --reference-image {moving_img} \
                --transform {mat} \
                --output {moco_img}'
        subprocess.run(cmd, shell=True)

    n_echo = len(files[con])

    for echo in range(1,n_echo+1):
        file_in  = denoising_dir /      f'den_{con}_e{echo}_m.nii'
        file_out = moco_dir      / f'moco_den_{con}_e{echo}_m.nii'
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

mt0 = rms_dir     /      'den_mt0_rms.nii'
mtw = moco_dir    / 'moco_den_mtw_rms.nii'
pdw = moco_dir    / 'moco_den_pdw_rms.nii'
t1w = moco_dir    / 'moco_den_t1w_rms.nii'
mpf = results_dir / 'mpf.nii'
t1f = results_dir / 't1f.nii'
r1f = results_dir / 'r1f.nii'

if t1f.exists():
    logger.info(f'fit-JSPqMT done: {t1f}')
else:
    d = JSPqMT['MTw_TIMINGS'] # shortcut
    MTW_TIMINGS: str  = f"{d['SatPulseDuration']},{d['Interdelay']},{d['ReadoutPulseDuration']},{d['TR']}"
    d = JSPqMT['VFA_TIMINGS'] # shortcut
    VFA_TIMINGS: str  = f"{d['ReadoutPulseDuration']},{d['TR']}"
    d = JSPqMT['MTw_PARX'] # shortcut
    MTW_PARX: str  = f"{d['ReadoutFA']},{d['ReadoutPulseShape']},{d['SaturationFA']},{d['SaturationOffset']},{d['SaturationPulseShape']}"
    d = JSPqMT['VFA_PARX'] # shortcut
    VFA_PARX: str  = f"{','.join(map(str,d['ReadoutFA']))},{d['ReadoutPulseShape']}"

    cmd = f'fit-JSPqMT {mt0},{mtw} {pdw},{t1w} {mpf} {t1f} \
            --R1f {r1f} \
            --mask {target_msk} \
            --B1 {b1map_smoothed} \
            --MTw_TIMINGS {MTW_TIMINGS} \
            --VFA_TIMINGS {VFA_TIMINGS} \
            --MTw_PARX {MTW_PARX} \
            --VFA_PARX {VFA_PARX} \
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
    with np.errstate(divide='ignore',invalid='ignore'): b = np.log(data_masked) # [ne, nvox]
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
    with np.errstate(divide='ignore',invalid='ignore'): t2s_flat = 1/r2s_flat
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
    with np.errstate(divide='ignore',invalid='ignore'): b = np.log(b)
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


def fit_t2s_arlo(data: np.ndarray,
                 TE  : np.ndarray,
                 mask: np.ndarray,
                 ) -> tuple[np.ndarray, np.ndarray] :
    
    nx,ny,nz,ne = data.shape
    dTE = np.diff(TE).mean()

    # in the paper, J=2, it means take echos #0 #1 #2, so a triplet
    nt = ne - 2 # numer of triplets
    Si = np.zeros((nx,ny,nz,nt), dtype=np.float32)
    Di = np.zeros((nx,ny,nz,nt), dtype=np.float32)

    for t in range(nt):
        Si[:,:,:,t] = (data[:,:,:,t] + 4*data[:,:,:,t+1] + data[:,:,:,t+2]) * dTE/3
        Di[:,:,:,t] =  data[:,:,:,t]                     - data[:,:,:,t+2]

    with np.errstate(divide='ignore',invalid='ignore'): 
        t2s = (        np.sum(Si*Si,axis=-1) + dTE/3 * np.sum(Si*Di,axis=-1)) \
            / (dTE/3 * np.sum(Si*Si,axis=-1) +         np.sum(Si*Di,axis=-1))
        
    t2s *= mask
    with np.errstate(divide='ignore',invalid='ignore'): r2s = 1/t2s
    t2s[t2s<0]    = 0
    t2s[t2s>1]    = 0
    r2s[r2s<0]    = 0
    r2s[r2s>1000] = 0
    return t2s, r2s


#%% fit R2* // ESTATICS

r2s_estatics = results_dir / f'r2s_estatics.nii'
if r2s_estatics.exists():
    logger.info(f'Already exist: {r2s_estatics}')
else:
    logger.info(f'R2s fit with ESTATICS method (multi contrast)')
    mask = nib.load(target_msk).get_fdata()

    imgs = MPMContainer()
    data = MPMContainer()
    te   = MPMContainer()
    imgs.mt0.m = [nib.load(file.root / file. den_nii) for file in files.mt0.m]
    imgs.mtw.m = [nib.load(file.root / file.moco_nii) for file in files.mtw.m]
    imgs.pdw.m = [nib.load(file.root / file.moco_nii) for file in files.pdw.m]
    imgs.t1w.m = [nib.load(file.root / file.moco_nii) for file in files.t1w.m]
    data.mt0.m = np.stack([img.get_fdata() for img in imgs.mt0.m], axis=3)
    data.mtw.m = np.stack([img.get_fdata() for img in imgs.mtw.m], axis=3)
    data.pdw.m = np.stack([img.get_fdata() for img in imgs.pdw.m], axis=3)
    data.t1w.m = np.stack([img.get_fdata() for img in imgs.t1w.m], axis=3)
    te.mt0.m = np.array([file.echo_time for file in files.mt0.m])
    te.mtw.m = np.array([file.echo_time for file in files.mtw.m])
    te.pdw.m = np.array([file.echo_time for file in files.pdw.m])
    te.t1w.m = np.array([file.echo_time for file in files.t1w.m])
    affine = imgs.mtw.m[0].affine
    header = imgs.mtw.m[0].header

    s0_mt0, s0_mtw, s0_pdw, s0_t1w, t2s, r2s, err = fit_t2s_estatics(
        data_mt0=data.mt0.m, data_mtw=data.mtw.m, data_pdw=data.pdw.m, data_t1w=data.t1w.m,
          TE_mt0=  te.mt0.m,   TE_mtw=  te.mtw.m,   TE_pdw=  te.pdw.m,   TE_t1w=  te.t1w.m,
        mask=mask
        )
    
    nib.save(nib.Nifti1Image(dataobj=r2s   , affine=affine, header=header, dtype=np.float32), results_dir / f'r2s_estatics.nii')
    nib.save(nib.Nifti1Image(dataobj=t2s   , affine=affine, header=header, dtype=np.float32), results_dir / f't2s_estatics.nii')
    nib.save(nib.Nifti1Image(dataobj=s0_mt0, affine=affine, header=header, dtype=np.float32), r2s_dir     / f's0_mt0_estatics.nii')
    nib.save(nib.Nifti1Image(dataobj=s0_mtw, affine=affine, header=header, dtype=np.float32), r2s_dir     / f's0_mtw_estatics.nii')
    nib.save(nib.Nifti1Image(dataobj=s0_pdw, affine=affine, header=header, dtype=np.float32), r2s_dir     / f's0_pdw_estatics.nii')
    nib.save(nib.Nifti1Image(dataobj=s0_t1w, affine=affine, header=header, dtype=np.float32), r2s_dir     / f's0_t1w_estatics.nii')

    logger.info(f'Wrote: {r2s_estatics}')


#%% fit R2* // OLS + WLS + NumART + ARLO on all contrasts, just to compare

for con in Contrast:

    r2s_ols = r2s_dir / f'r2s_{con}_ols.nii'
    if r2s_ols.exists():
        logger.info(f'Already exist: {r2s_ols}')
    else:
        logger.info(f'R2s **single contrast** fits for {con}')
        mask = nib.load(target_msk).get_fdata()

        if   con is Contrast.mt0: imgs = [nib.load(file.root / file. den_nii) for file in files.mt0.m]; te = np.array([file.echo_time for file in files.mt0.m])
        elif con is Contrast.mtw: imgs = [nib.load(file.root / file.moco_nii) for file in files.mtw.m]; te = np.array([file.echo_time for file in files.mtw.m])
        elif con is Contrast.pdw: imgs = [nib.load(file.root / file.moco_nii) for file in files.pdw.m]; te = np.array([file.echo_time for file in files.pdw.m])
        elif con is Contrast.t1w: imgs = [nib.load(file.root / file.moco_nii) for file in files.t1w.m]; te = np.array([file.echo_time for file in files.t1w.m])
        data = np.stack([img.get_fdata() for img in imgs], axis=3)

        s0, t2s, r2s, err = fit_t2s_loglin_leastsquare(data, te, mask, 'ols')
        nib.save(nib.Nifti1Image(dataobj=r2s, affine=imgs[0].affine, header=imgs[0].header, dtype=np.float32), r2s_dir / f'r2s_{con}_ols.nii')
        nib.save(nib.Nifti1Image(dataobj=t2s, affine=imgs[0].affine, header=imgs[0].header, dtype=np.float32), r2s_dir / f't2s_{con}_ols.nii')
        nib.save(nib.Nifti1Image(dataobj=s0 , affine=imgs[0].affine, header=imgs[0].header, dtype=np.float32), r2s_dir /  f's0_{con}_ols.nii')

        s0, t2s, r2s, err = fit_t2s_loglin_leastsquare(data, te, mask, 'wls')
        nib.save(nib.Nifti1Image(dataobj=r2s, affine=imgs[0].affine, header=imgs[0].header, dtype=np.float32), r2s_dir / f'r2s_{con}_wls.nii')
        nib.save(nib.Nifti1Image(dataobj=t2s, affine=imgs[0].affine, header=imgs[0].header, dtype=np.float32), r2s_dir / f't2s_{con}_wls.nii')
        nib.save(nib.Nifti1Image(dataobj=s0 , affine=imgs[0].affine, header=imgs[0].header, dtype=np.float32), r2s_dir /  f's0_{con}_wls.nii')

        with np.errstate(divide='ignore',invalid='ignore'): 
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

        t2s, r2s = fit_t2s_arlo(data, te, mask)
        nib.save(nib.Nifti1Image(dataobj=r2s, affine=imgs[0].affine, header=imgs[0].header, dtype=np.float32), r2s_dir / f'r2s_{con}_arlo.nii')
        nib.save(nib.Nifti1Image(dataobj=t2s, affine=imgs[0].affine, header=imgs[0].header, dtype=np.float32), r2s_dir / f't2s_{con}_arlo.nii')
        logger.info(f'Wrote on disk all R2s **single contrast** fits for {con}')

