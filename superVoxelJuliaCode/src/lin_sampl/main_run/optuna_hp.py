# sudo chmod 777 /media/jm/hddData/datasets/mysql_hp

# docker stop mysql-hp
# docker rm mysql-hp




#   # -e MYSQL_PASSWORD=hppass \


# docker run -d \
#   --name mysql-hp \
#   --restart unless-stopped \
#   -e MYSQL_ROOT_PASSWORD=mysecret \
#   -e MYSQL_DATABASE=hp \
#   -e MYSQL_USER=hpuser \
#   -v /media/jm/hddData/datasets/mysql_hp:/var/lib/mysql \
#   -p 3306:3306 \
#   mysql:8.0


# docker exec -it mysql-hp mysql -u root -p'mysecret'

# CREATE USER 'hpuser'@'%' IDENTIFIED BY 'mysecret';
# GRANT ALL PRIVILEGES ON hp.* TO 'hpuser'@'%';
# FLUSH PRIVILEGES;

# docker network create hp-net
# docker network connect hp-net mysql-hp

# #check connection
# docker exec -it mysql-hp mysql -u hpuser -h 127.0.0.1 -p'mysecret' hp



import importlib.util
import sys
import optuna
# from ray import air, tune
# from ray.air import session
# from ray.tune import CLIReporter
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice
# torch.multiprocessing.freeze_support()

from sqlalchemy import URL
import sqlalchemy,os
import subprocess,pandas as pd
import tempfile

# Install driver if missing: pip install PyMySQL

# engine = sqlalchemy.create_engine("mysql+pymysql://hpuser@127.0.0.1:3306/hp")
# with engine.connect() as connection:
#     result = connection.execute("SELECT 1;")
#     print(result.fetchone())
    
    
# url_object = URL.create(
#     drivername="mysql+pymysql",
#     username="hpuser",
#     password="hppass",
#     host="127.0.0.1",
#     port=3306,
#     database="hp"
# )

# str(url_object)

experiment_name="svox_4"

study = optuna.create_study(
    study_name=experiment_name,
    sampler=optuna.samplers.NSGAIISampler(),
    pruner=optuna.pruners.HyperbandPruner(),
    storage="mysql+pymysql://hpuser:mysecret@127.0.0.1:3306/hp",
    load_if_exists=True,
    direction="minimize",
)


def run_julia_command(trial,out_path,restarted):
    # Build command with trial values
    display = os.environ.get('DISPLAY', ':0')
    dev = os.environ.get('devv', '1')
    print(f"dddddddddd  {dev}")
    # str(trial.suggest_categorical("add_gradient_accum", [True, False])).lower()
    grad_accum=trial.suggest_categorical("grad_accum_val", [0,2, 6, 10])
    clipp=trial.suggest_categorical("clip_norm_val", [0,1.5, 10.0, 100.0])
    
    cmd = f"""DISPLAY={display}
        LEARNING_RATE_START={trial.suggest_float("learning_rate_start", 0.00000001, 0.01, log=True)} \
        LEARNING_RATE_END={trial.suggest_float("learning_rate_end", 0.00000001, 0.001, log=True)} \
        ADD_GRADIENT_ACCUM={str(grad_accum>0).lower()} \
        ADD_GRADIENT_NORM={str(clipp>0).lower()} \
        IS_WEIGHT_DECAY={str(trial.suggest_categorical("is_WeightDecay", [True, False])).lower()} \
        GRAD_ACCUM_VAL={grad_accum} \
        CLIP_NORM_VAL={clipp} \
        out_path={out_path} \
        CUDA_VISIBLE_DEVICES={dev} \
        restart={int(restarted)} \
        julia /workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/lin_sampl_main_run.jl"""

      # Execute command
    result = subprocess.run(cmd, 
                        shell=True, 
                        check=True,
                        capture_output=True,
                        text=True)

    # Access results if needed
    # print(f"Return code: {result.returncode}")
    # print(f"Output: {result.stdout}")
    print(f"eeeee Errors: {result.stderr}")

"""
tries to execute julia code and returns True if it was successful, False otherwise
"""
def run_juliaa(trial, out_path,restarted,csv_path):
  try:
    run_julia_command(trial, out_path,restarted)
    if not os.path.exists(csv_path):
      return False
    return True
  except Exception as e:
    print(f"Error: {e}")
    return False
    
def objective(trial):
  # Create a temporary directory
  with tempfile.TemporaryDirectory() as tmpdirname:
      out_path = tmpdirname
      csv_path=f"{out_path}/res.csv"
      success=run_juliaa(trial, out_path,False,csv_path)
      while not success:
          success = run_juliaa(trial, out_path, True, csv_path)
          

      df = pd.read_csv(csv_path)
      res_value = df.loc[0, "res"]
      

      return res_value
  
  
  run_julia_command(trial,out_path)
  df = pd.read_csv(f"out_path/res.csv")
  res_value = df.loc[0, "res"]
  

  
study.optimize(objective, n_trials=1000)  
  
  



  

# devv=0 python3 /workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/optuna_hp.py
# devv=1 python3 /workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/optuna_hp.py





