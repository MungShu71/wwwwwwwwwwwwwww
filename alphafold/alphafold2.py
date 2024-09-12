import os
import re
import hashlib
import random
import argparse
import numpy
import sys
import warnings
from Bio import BiopythonDeprecationWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=BiopythonDeprecationWarning)
from pathlib import Path
import matplotlib.pyplot as plt

from colabfold.ColabFold.colabfold.download import download_alphafold_params
from colabfold.ColabFold.colabfold.utils import setup_logging
from colabfold.ColabFold.colabfold.batch import get_queries, run, set_model_type
from colabfold.ColabFold.colabfold.plot import plot_msa_v2
from colabfold.ColabFold.colabfold.colabfold import plot_protein

from sys import version_info
python_version = f"{version_info.major}.{version_info.minor}"

# hash function 
def add_hash(x,y):
  return x+"_"+hashlib.sha1(y.encode()).hexdigest()[:5]

parser = argparse.ArgumentParser(prog = "alphafold2.py", add_help = True)

parser.add_argument("--query_seq")
parser.add_argument("--job_name")

parser.add_argument("--num_relax", 
                    type = int, 
                    choices = [0, 1, 5], 
                    default = 0)

parser.add_argument("--template_mode", 
                    type = str, 
                    choices = ["none", "pdb100", "custom"], 
                    default = "none" )

parser.add_argument("--msa_mode", 
                    type = str, 
                    choices = ["mmseqs2_uniref_env", "mmseqs2_uniref","single_sequence","custom"],
                    default = "mmseqs2_uniref_env")

parser.add_argument("--pair_mode",
                    type = str,
                    choices = ["unpaired_paired","paired","unpaired"],
                    default = "unpaired_paired")

parser.add_argument("--model_type",
                    type = str,
                    choices = ["auto", "alphafold2_ptm", "alphafold2_multimer_v1", "alphafold2_multimer_v2", "alphafold2_multimer_v3", "deepfold_v1"],
                    default = "auto")

parser.add_argument("--num_recycles",
                    type = str,
                    choices = ["auto", "0", "1", "3", "6", "12", "24", "48"],
                    default = "3")

parser.add_argument("--recycle_early_stop_tolerance",
                    type = str,
                    choices = ["auto", "0.0", "0.5", "1.0"],
                    default = "auto")

parser.add_argument("--relax_max_iterations",
                    type = int,
                    choices = [0, 200, 2000],
                    default = 200)

parser.add_argument("--pairing_strategy",
                    type = str,
                    choices = ["greedy", "complete"], 
                    default = "greedy")

parser.add_argument("--max_msa",
                    type = str,
                    choices = ["auto", "512:1024", "256:512", "64:128", "32:64", "16:32"],
                    default = "auto")

parser.add_argument("--num_seeds",
                    type = int,
                    choices = [1, 2, 4, 8, 16],
                    default = 1)

parser.add_argument("--use_dropout",
                    type = bool,
                    default = True)

parser.add_argument("--dpi",
                    type = int,
                    default = 200)

args = vars(parser.parse_args())

query_seq = args["query_seq"]
jobname = args["job_name"]
num_relax = args["num_relax"]
template_mode = args["template_mode"]
msa_mode = args["msa_mode"]
pair_mode = args["pair_mode"]

#if `auto` selected, will use `alphafold2_ptm` for monomer prediction and 
# `alphafold2_multimer_v3` for complex prediction.
# ????
model_type = args["model_type"] 

num_recycles = args["num_recycles"]
recycle_early_stop_tolerance = args["recycle_early_stop_tolerance"]
relax_max_iterations = args["relax_max_iterations"]
pairing_strategy = args["pairing_strategy"]
max_msa = args["max_msa"]
num_seeds = args["num_seeds"]
use_dropout = args["use_dropout"]
dpi = args["dpi"]

display_images = True
save_all = False 
save_recycles = False 
save_to_google_drive = False


num_recycles = None if num_recycles == "auto" else int(num_recycles)
recycle_early_stop_tolerance = None if recycle_early_stop_tolerance == "auto" else float(recycle_early_stop_tolerance)
if max_msa == "auto": max_msa = None

'''
if num_recycles == "auto" and model_type == "alphafold2_multimer_v3":
  num_recycles = 20

if recycle_early_stop_tolerance == "auto":
    if model_type == "alphafold2_multimer_v3":
       recycle_early_stop_tolerance = 0.5
    else:
       recycle_early_stop_tolerance = 0.0
'''

use_amber = num_relax > 0

# # remove whitespaces
# query_seq = query_seq.strip()
# basejobname = jobname.strip()

jobname = add_hash(jobname, query_seq)

def check(folder):
    return not os.path.exists(folder)
# # check if there are multiple files with same job names
# # prevents data overwriting
if not check(jobname):
   n = 0
   while not check(f"{jobname}_{n}"):
      n += 1
   jobname = f"{jobname}_{n}"
    

# make directory for results
os.makedirs(jobname, exist_ok = True)

# save queries
queries_path = os.path.join(jobname, f"{jobname}.csv")
with open (queries_path, "w") as file:
   file.write(f"id, sequence\n{jobname}, {query_seq}")

match template_mode:
   case "pdb100":
      use_templates = True
      custom_template_path = None
   case "custom":
      custom_template_path = os.path.join(jobname, f"template")
      os.makedirs(custom_template_path, exist_ok = True)
      """ 
      uploaded_files = files.upload() 
      for file in uploaded_files:
         os.rename(file, os.path.join(custom_template_path, file))
      """
      use_templates = True
   case _:
      custom_template_path = None
      use_templates = False


USE_AMBER = use_amber
USE_TEMPLATES = use_templates
PYTHON_VERSION = python_version

if not os.path.isfile("COLABFOLD_READY"):
  print("installing colabfold...")
  os.system("pip install -q --no-warn-conflicts 'colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold'")
  if os.environ.get('TPU_NAME', False) != False:
    os.system("pip uninstall -y jax jaxlib")
    os.system("pip install --no-warn-conflicts --upgrade dm-haiku==0.0.10 'jax[cuda12_pip]'==0.3.25 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
  os.system("ln -s /usr/local/lib/python3.*/dist-packages/colabfold colabfold")
  os.system("ln -s /usr/local/lib/python3.*/dist-packages/alphafold alphafold")
  os.system("touch COLABFOLD_READY")

if USE_AMBER or USE_TEMPLATES:
  if not os.path.isfile("CONDA_READY"):
    print("installing conda...")
    os.system("wget -qnc https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh")
    os.system("bash Mambaforge-Linux-x86_64.sh -bfp /usr/local")
    os.system("mamba config --set auto_update_conda false")
    os.system("touch CONDA_READY")

if USE_TEMPLATES and not os.path.isfile("HH_READY") and USE_AMBER and not os.path.isfile("AMBER_READY"):
  print("installing hhsuite and amber...")
  os.system(f"mamba install -y -c conda-forge -c bioconda kalign2=2.04 hhsuite=3.3.0 openmm=7.7.0 python='{PYTHON_VERSION}' pdbfixer")
  os.system("touch HH_READY")
  os.system("touch AMBER_READY")
else:
  if USE_TEMPLATES and not os.path.isfile("HH_READY"):
    print("installing hhsuite...")
    os.system(f"mamba install -y -c conda-forge -c bioconda kalign2=2.04 hhsuite=3.3.0 python='{PYTHON_VERSION}'")
    os.system("touch HH_READY")
  if USE_AMBER and not os.path.isfile("AMBER_READY"):
    print("installing amber...")
    os.system(f"mamba install -y -c conda-forge openmm=7.7.0 python='{PYTHON_VERSION}' pdbfixer")
    os.system("touch AMBER_READY")


if "mmseq2" in msa_mode:
   a3m_file = os.path.join(jobname, f"{jobname}.a3m")
elif msa_mode == "custom":
   a3m_file = os.path.join(jobname, f"{jobname}.custom.a3m")
   

if use_amber and f"/usr/local/lib/python{python_version}/site-packages/" not in sys.path:
    sys.path.insert(0, f"/usr/local/lib/python{python_version}/site-packages/")


def input_features_callback(input_features):
  if display_images:
    plot_msa_v2(input_features)
    plt.show()
    plt.close()

def prediction_callback(protein_obj, length,
                        prediction_result, input_features, mode):
  model_name, relaxed = mode
  if not relaxed:
    if display_images:
      fig = plot_protein(protein_obj, Ls=length, dpi=150)
      plt.show()
      plt.close()


result_dir = jobname
log_filename = os.path.join(jobname,"log.txt")
setup_logging(Path(log_filename))

queries, is_complex = get_queries(queries_path)
model_type = set_model_type(is_complex, model_type)

queries, is_complex = get_queries(queries_path)
model_type = set_model_type(is_complex, model_type)

if "multimer" in model_type and max_msa is not None:
  use_cluster_profile = False
else:
  use_cluster_profile = True

download_alphafold_params(model_type, Path("."))
results = run(
    queries=queries,
    result_dir=result_dir,
    use_templates=use_templates,
    custom_template_path=custom_template_path,
    num_relax=num_relax,
    msa_mode=msa_mode,
    model_type=model_type,
    num_models=5,
    num_recycles=num_recycles,
    relax_max_iterations=relax_max_iterations,
    recycle_early_stop_tolerance=recycle_early_stop_tolerance,
    num_seeds=num_seeds,
    use_dropout=use_dropout,
    model_order=[1,2,3,4,5],
    is_complex=is_complex,
    data_dir=Path("."),
    keep_existing_results=False,
    rank_by="auto",
    pair_mode=pair_mode,
    pairing_strategy=pairing_strategy,
    stop_at_score=float(100),
    prediction_callback=prediction_callback,
    dpi=dpi,
    zip_results=False,
    save_all=save_all,
    max_msa=max_msa,
    use_cluster_profile=use_cluster_profile,
    input_features_callback=input_features_callback,
    save_recycles=save_recycles,
    user_agent="colabfold/google-colab-main",
)
results_zip = f"{jobname}.result.zip"
os.system(f"zip -r {results_zip} {jobname}")