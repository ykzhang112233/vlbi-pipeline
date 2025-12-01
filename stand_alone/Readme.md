## Header
- The vlbi pipeline script used for phase-referencing observations.
- This pipeline uses the `ParselTongue` package for python scripting the `AIPS` prcessings.

## Pre-requisites
- AIPS installed and configured.
- ParselTongue installed.
- Obit installed
- Python 2.7 or 3.x. (python version should be in agreement with the ParselTongue version)

## Main steps of the pipeline
Part I: Initial calibration
1. Data import and indexation. (fitld and indxr)
2. inosphereic TEC correction and Earth Orientation Parameters (EOP) correction.
3. A priori amplitude calibration.(accor, apcal)
4. Parrallactic angle correction.
5. Automated flagging of bad data. (aclip, quack etc.)
6. Data inspection plots. (possom, snplt ,vplot etc.)
    After this part, mannual data inspection and flagging is recommended. Select a good scan for mannual phase calibration in the next part. Select the reference antenna for next parts.
Part II: Main fringe-fitting and calibration
7. (optional) Additional amplitede calibration using auto-correlation and bandpass calibration. (acscl, bpass) # Note: this is newer method, and mostly for complex bandpass and heavy RFI bands such as L-band.
8. Mannual phase calibration using fringe finder sources. (fring)
9. Main fringe-fitting using phase-referencing calibrators. (fring)
10. Bandpass calibration using bright sources. (bpass)
11. Solution application and phase calibrator imaging. (clcal, split)
    At the end of this part, mannual inspection of the calibrator images and solutions is recommended. 
    Do imaging and self-calibration on the phase calibrator to improve the model in Difmap and generate a new clean component model file for the next step.
    Also, note down the antenna gain correction factors during self-calibration for later use.
    If necessary, repeat the fringe-fitting with different parameters or flag more bad data.
Part III: Target source imaging and analysis
12. (optional) Apply antenna gain correction factors obtained from self-calibration to the phase calibrator and target sources. (clcor)
12. Read in the new clean component model file of the phase calibrator.
13. Re-fringe fitting of the phase calibrator using the new model. (fring)
14. Solution application and target source imaging (optional: IF averaging). (clcal, split)
15. Target source imaging and analysis in Difmap or AIPS
16. (optional) Astrometric adjustments (uvshift). # Note: only when the target source is found to be far away from the phase center after first imaging. It is designed to deal with the smearing effect.

## How to use it
The data should be first downloaded to the data directory and assign it in the `ba161a-pip-parms.py` input parm.
1. Modify the input parameters in the parm file according to your data and needs.
    1.1 Set the data path and name in the `Input Parameters part I` of the parm file. The set the control flags `step1=1` (the rest should be 0) and then run the pipeline to do initial data import and inspection.
    1.2 After the first part ends, inpsect the data according to the output file and set parameters in the `Input Parameters part II` of the parm file. Then run using `step2=1` (the rest should be 0) to do the main calibration.
    1.3 After the second part ends, image the phase calibrator and self-calibrate it in Difmap. Note down the gain correction factors if any.
         Set the parameters in the `Input Parameters part III` of the parm file. Then run using `step3=1` (the rest should be 0) to do the target source imaging.

2. Run the pipeline script using the command:
   ```shell
   parseltongue ba161a-pipeline.py
   ```
Note for ba161: out input data have include the parameters for all three steps, so one can reproduce the whole process by running the pipeline just once with step1=1, step2=1 and step3=1.
Usually, the `SCL10` version is recommended for final analysis based on the standard calibration process.
