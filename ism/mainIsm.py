
# MAIN FUNCTION TO CALL THE ISM MODULE

from ism.src.ism import ism

# Directory - this is the common directory for the execution of the E2E, all modules
auxdir = r'C:/GitHub/EODP_repo_Jaime/auxiliary'
indir = r"C:/MasterEODP/EODP_TER_2021/EODP-TS-ISM/input/gradient_alt100_act150" # small scene
outdir = r"C:/MasterEODP/EODP_TER_2021/EODP-TS-ISM/myOutput"

# Initialise the ISM
myIsm = ism(auxdir, indir, outdir)
myIsm.processModule()
