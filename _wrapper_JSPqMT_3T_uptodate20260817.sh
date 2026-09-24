#!/bin/bash
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=24
CONDA_ENV_qMT=qMT
CONDA_ENV_tmppca=tmppca

##################################################
##################################################
# PATHNAMES ######################################
##################################################
##################################################
ROOT_PATH=/home/lsoustelle/Desktop/DATA/DEV_tMPPCA/
NII_FLD=${ROOT_PATH}/data_qMT/
TMP_FLD=${ROOT_PATH}/tmp/
PROC_FLD=${ROOT_PATH}/proc/
DEN_ARG="--MPPCA --N_SoS 1"
mkdir -p ${TMP_FLD} ${PROC_FLD} ${NII_FLD}

######################################## PATHS VFA/MTw/ANAT/B1
B1RAW_NII=${NII_FLD}/6_tfl_b1map_sag.nii
B1_FAC=800 ## presat-TFL: 800 for Vida, 900 for Terra
ANAT_NII=${NII_FLD}/27_vibeMT_t1w_sag_6eco_1iso_c9.nii
LIST_PREF=( "33_vibeMT_pdw_sag_6eco_1iso_c9" 
			"29_vibeMT_mt0_sag_6eco_1iso_c9" 
			"27_vibeMT_t1w_sag_6eco_1iso_c9" 
			"9_vibeMT_mtw_sag_3eco_1iso_c9" )

B1map_IN_ANAT_NII=${PROC_FLD}/B1_MAP_inANAT.nii.gz
ANAT_MASKED_NII=${TMP_FLD}/ANAT_DENN4_MASKED.nii.gz
ANAT_DENN4_OUTNII=${PROC_FLD}/ANAT_denN4.nii.gz
ANAT_DENN4mask_OUTNII=${TMP_FLD}/ANAT_denN4_.nii.gz
MASK_ANAT_NII=${PROC_FLD}/ANAT_MASK.nii.gz
########################################

#################################################
#################################################
# ANAT ##########################################
#################################################
#################################################
## Den/N4, mask
if [[ ! -f ${ANAT_DENN4_OUTNII} ]]; then
	ImageMath 4 ${TMP_FLD}/split_.nii.gz TimeSeriesDisassemble ${ANAT_NII}
	DenoiseImage -d 3 -i ${TMP_FLD}/split_1000.nii.gz -o ${ANAT_DENN4_OUTNII} -v 1
	rm ${TMP_FLD}/split_100*
	N4BiasFieldCorrection -d 3 -i ${ANAT_DENN4_OUTNII} -o ${ANAT_DENN4_OUTNII} -c [50x50x50x50,0.00001] -v 1
fi
if [[ ! -f ${ANAT_MASKED_NII} ]]; then
	mri_synthstrip -i ${ANAT_DENN4_OUTNII} -m ${MASK_ANAT_NII}
	ImageMath 3 ${ANAT_MASKED_NII} m ${MASK_ANAT_NII} ${ANAT_DENN4_OUTNII}
fi

##################################################
##################################################
# B1 map #########################################
##################################################
##################################################
if [[ ! -f ${B1map_IN_ANAT_NII} ]]; then
	ImageMath 3 ${B1map_IN_ANAT_NII} / ${B1RAW_NII} ${B1_FAC} 
	antsApplyTransforms -d 3 -v 1 \
						-i ${B1map_IN_ANAT_NII} \
						-o ${B1map_IN_ANAT_NII} \
						-r ${ANAT_DENN4_OUTNII}
	ImageMath 3 ${B1map_IN_ANAT_NII} G ${B1map_IN_ANAT_NII} 3
	ImageMath 3 ${B1map_IN_ANAT_NII} m ${B1map_IN_ANAT_NII} ${MASK_ANAT_NII}
fi

##################################################
##################################################
# MPPCA, SoS & MoCo VFA/MTw IMAGES ###############
##################################################
##################################################
LIST_NII=()
for ((ii=0; ii < ${#LIST_PREF[@]}; ii++ )); do
	LIST_NII=(${LIST_NII[@]} `echo ${NII_FLD}/*${LIST_PREF[$ii]}*.nii*`)
done
# printf "%s\n" "${LIST_NII[@]}"; exit

# FNAMES
LIST_MASKS_NII=(${LIST_NII[@]/.nii/_mask.nii.gz})
LIST_MASKS_NII=(${LIST_MASKS_NII[@]/${NII_FLD}/${TMP_FLD}})
LIST_MPPCA_SOS_NII=(${LIST_NII[@]/.nii/_denMPPCA_SOS.nii.gz})
LIST_MPPCA_SOS_NII=(${LIST_MPPCA_SOS_NII[@]/${NII_FLD}/${TMP_FLD}})
LIST_MPPCA_SOS_N4BER_NII=(${LIST_MPPCA_SOS_NII[@]/.nii.gz/_N4BER.nii.gz}) # BER=Brain Extracted
LIST_MPPCA_SOS_N4BER_MoCoed_NII=(${LIST_MPPCA_SOS_N4BER_NII[@]/.nii.gz/_MoCoed.nii.gz})
LIST_MPPCA_SOS_BER_MoCoed_NII=(${LIST_MPPCA_SOS_NII[@]/.nii.gz/_BER_MoCoed_Warped.nii.gz}) # after apply transform
LIST_MPPCA_SOS_BER_MoCoedinANAT_NII=(${LIST_MPPCA_SOS_BER_MoCoed_NII[@]/.nii.gz/_inANAT.nii.gz}) # after apply transform

### Perform MPPCA & SoS (switch conda env.)
# echo ${LIST_MPPCA_SOS_NII[0]}
# echo $(echo ${LIST_NII[@]} | tr ' ' ,)
if [[ ! -f ${LIST_MPPCA_SOS_NII[0]} ]]; then
	if [[ $CONDA_DEFAULT_ENV != $CONDA_ENV_tmppca ]]; then
		eval "$(conda shell.bash hook)"
		conda activate $CONDA_ENV_tmppca
	fi
	python3 wrapper_MPPCA_SoS_CLI.py 	$(echo ${LIST_NII[@]} | tr ' ' ,) \
										--SUFFIX_NII _denMPPCA_SOS.nii.gz --OUT_DIR ${TMP_FLD}/ $DEN_ARG \
										--nthreads $ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS
	if [[ $CONDA_DEFAULT_ENV != $CONDA_ENV_qMT ]]; then
		eval "$(conda shell.bash hook)"
		conda activate $CONDA_ENV_qMT
	fi
fi

### Generate brain mask for all volumes
for ((ii=0; ii < ${#LIST_MASKS_NII[@]}; ii++ )); do
	if [[ ! -f ${LIST_MASKS_NII[$ii]} ]]; then
		mri_synthstrip -i ${LIST_MPPCA_SOS_NII[$ii]} -m ${LIST_MASKS_NII[$ii]}
	fi
done

### N4+BER all VFA/MTw images
if [[ ! -f ${TMP_FLD}/BiasField_T1wVFA.nii.gz ]]; then
	N4BiasFieldCorrection 	-d 3 -c [50x50x50x50,0.00001] -v 1 \
							-i ${LIST_MPPCA_SOS_NII[2]} \
							-o [${LIST_MPPCA_SOS_NII[2]/.nii.gz/_N4.nii.gz},${TMP_FLD}/BiasField_T1wVFA.nii.gz]
fi
for ((ii=0; ii < ${#LIST_MPPCA_SOS_N4BER_NII[@]}; ii++ )); do
	if [[ ! -f ${LIST_MPPCA_SOS_N4BER_NII[$ii]} ]]; then
		ImageMath 3 ${LIST_MPPCA_SOS_N4BER_NII[$ii]} / ${LIST_MPPCA_SOS_NII[$ii]} ${TMP_FLD}/BiasField_T1wVFA.nii.gz
		ImageMath 3 ${LIST_MPPCA_SOS_N4BER_NII[$ii]} m ${LIST_MPPCA_SOS_N4BER_NII[$ii]} ${LIST_MASKS_NII[$ii]}
	fi
done

### MoCo: VFA/MTw N4-BER on T1w-VFA (no initial movement), and apply transform on non-N4
for ((ii=0; ii < ${#LIST_MPPCA_SOS_N4BER_NII[@]}; ii++)); do
	# estimate MoCo transfo on N4 images
	if [[ ! -f ${LIST_MPPCA_SOS_N4BER_MoCoed_NII[$ii]/.nii.gz/_Warped.nii.gz} ]]; then
		antsRegistration 	-d 3 -v 1 --float 0 \
							--output [${LIST_MPPCA_SOS_N4BER_MoCoed_NII[$ii]/.nii.gz/_},${LIST_MPPCA_SOS_N4BER_MoCoed_NII[$ii]/.nii.gz/_Warped.nii.gz}] \
							--interpolation Linear \
							--winsorize-image-intensities [0.005,0.995] \
							--use-histogram-matching 0 \
							--transform Rigid[0.1] \
							--metric MI[${LIST_MPPCA_SOS_N4BER_NII[2]},${LIST_MPPCA_SOS_N4BER_NII[$ii]},1,32,Regular,0.25] \
							--convergence [250x100,1e-6,10] \
							--shrink-factors 2x1 \
							--smoothing-sigmas 1x0vox
	fi
done

##################################################
##################################################
# Register MT/VFA to ANAT & cat ##################
##################################################
##################################################
### estimate transform T1w-VFA->ANAT
if [[ ! -f ${LIST_MPPCA_SOS_N4BER_MoCoed_NII[2]/.nii.gz/_WarpedtoAnat.nii.gz} ]]; then
	antsRegistration 	-d 3 -v 1 --float 0 \
						--output [${LIST_MPPCA_SOS_N4BER_MoCoed_NII[2]/.nii.gz/_toAnat_},${LIST_MPPCA_SOS_N4BER_MoCoed_NII[2]/.nii.gz/_WarpedtoAnat.nii.gz}] \
						--interpolation Linear \
						--use-histogram-matching 0 \
						--transform Rigid[0.1] \
						--metric MI[${ANAT_MASKED_NII},${LIST_MPPCA_SOS_N4BER_NII[2]},1,32,Regular,0.25] \
						--convergence [100,1e-6,10] \
						--shrink-factors 1 \
						--smoothing-sigmas 0vox
fi

### Apply transform All->T1w-VFA->ANAT on MPPCA-SOS images - single trf to avoid multiple interpolation
for ((ii=0; ii < ${#LIST_MPPCA_SOS_BER_MoCoedinANAT_NII[@]}; ii++ )); do
	if [[ ! -f ${LIST_MPPCA_SOS_BER_MoCoedinANAT_NII[$ii]} ]]; then
		antsApplyTransforms -d 3 -v 1 \
							-i ${LIST_MPPCA_SOS_NII[$ii]} \
							-o ${LIST_MPPCA_SOS_BER_MoCoedinANAT_NII[$ii]} \
							-t ${LIST_MPPCA_SOS_N4BER_MoCoed_NII[$ii]/.nii.gz/_0GenericAffine.mat} \
							-t ${LIST_MPPCA_SOS_N4BER_MoCoed_NII[2]/.nii.gz/_toAnat_0GenericAffine.mat} \
							-r ${ANAT_MASKED_NII}

		ImageMath 3 ${LIST_MPPCA_SOS_BER_MoCoedinANAT_NII[$ii]} \
					m \
					${LIST_MPPCA_SOS_BER_MoCoedinANAT_NII[$ii]} \
					${MASK_ANAT_NII}
	fi
done

### Concatenate for processing
LIST_MTw=( ${LIST_MPPCA_SOS_BER_MoCoedinANAT_NII[1]} ${LIST_MPPCA_SOS_BER_MoCoedinANAT_NII[3]} ) # only MT0/MTw
LIST_VFA=( ${LIST_MPPCA_SOS_BER_MoCoedinANAT_NII[0]} ${LIST_MPPCA_SOS_BER_MoCoedinANAT_NII[2]} ) # only PDw/T1w

if [[ ! -f ${PROC_FLD}/MTw_inANAT_denMPPCA.nii.gz ]]; then
	ImageMath 4 ${PROC_FLD}/MTw_inANAT_denMPPCA.nii.gz TimeSeriesAssemble 1 0 ${LIST_MTw[@]}
fi
if [[ ! -f ${PROC_FLD}/VFA_inANAT_denMPPCA.nii.gz ]]; then
	ImageMath 4 ${PROC_FLD}/VFA_inANAT_denMPPCA.nii.gz TimeSeriesAssemble 1 0 ${LIST_VFA[@]}
fi


##################################################
##################################################
# Joint Single-Point qMT #########################
##################################################
##################################################
## MPPCA+SOS
if [[ ! -f ${PROC_FLD}/MPF_JSPqMT_denMPPCA.nii.gz ]]; then
	fit-JSPqMT \
		${PROC_FLD}/MTw_inANAT_denMPPCA.nii.gz \
		${PROC_FLD}/VFA_inANAT_denMPPCA.nii.gz \
		${PROC_FLD}/MPF_JSPqMT_denMPPCA.nii.gz \
		${PROC_FLD}/T1f_JSPqMT_denMPPCA.nii.gz \
		--MTw_TIMINGS 12.0,2.1,0.5,30.0 \
		--VFA_TIMINGS 0.25,30.0 \
		--VFA_PARX 6.0,33.0,BP \
		--MTw_PARX 14.0,BP,560.0,4000.0,Hann-Sine \
		--qMTconstraint_PARX 0.0158,10.0e-6,21.1 \
		--use_GBM \
		--B1 ${B1map_IN_ANAT_NII} \
		--mask ${MASK_ANAT_NII} \
		--nworkers ${ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS} \
		--cpp_opt
fi


