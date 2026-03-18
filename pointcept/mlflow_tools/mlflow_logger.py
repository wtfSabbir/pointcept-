import mlflow
import os
import mlflow.data
import mlflow.pytorch
import pandas as pd
import glob
import numpy as np


class MLflowLogger:
    def __init__(self, tracking_uri, experiment_name, run_name):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        self.active_run = mlflow.start_run(run_name=run_name)
        print(f"\n[MLflow] Connected! Started Run: {run_name}")
        print(f"[MLflow] Run ID: {self.active_run.info.run_id}\n")

    def log_parameters(self, params_dict):
        try:
            mlflow.log_params(params_dict)
            print("[MLflow] Parameters logged successfully.")
        except Exception as e:
            print(f"[MLflow Error] Could not log parameters: {e}")

    def log_metric(self, key, value, step):
        try:
            mlflow.log_metric(key, float(value), step=step)
        except Exception as e:
            pass

    def log_artifact(self, file_path):
        if os.path.exists(file_path):
            mlflow.log_artifact(file_path)
            print(f"[MLflow] Saved artifact: {file_path}")
        else:
            print(f"[MLflow Warning] Tried to save {file_path}, but it doesn't exist.")

    def close(self):
        mlflow.end_run()
        print("[MLflow] Run closed and finalized.")
    

    
    def log_pytorch_model(self, model, model_name="Best_PTv3_Model"):
        try:
            
            # Step 1: Force legacy saving behavior using artifact_path
            mlflow.pytorch.log_model(
                pytorch_model=model, 
                artifact_path="model_package"
            )
            
            # Step 2: Manually link it to the Registry
            if mlflow.active_run() is not None:
                run_id = mlflow.active_run().info.run_id
                model_uri = f"runs:/{run_id}/model_package"
                mlflow.register_model(model_uri, model_name)
                print(f"[MLflow] Successfully registered {model_name} to the Models tab.")
        except Exception as e:
            print(f"[MLflow Warning] Could not log PyTorch model: {e}")
    
    def log_dataset(self, folder_path, dataset_name, context="training"):
        """Logs dataset metadata to the MLflow Datasets tab."""
        try:
 
            search_pattern = os.path.join(folder_path, "**", "*coord.npy")
            
            file_count = len(glob.glob(search_pattern, recursive=True))

            meta_df = pd.DataFrame({"True_Scans": [file_count], "Folder_Path": [folder_path]})

            dataset = mlflow.data.from_pandas(meta_df, source=folder_path, name=dataset_name)
            mlflow.log_input(dataset, context=context)
            
            print(f"[MLflow] Logged {dataset_name} ({file_count} true scans) to Datasets tab.")
        except Exception as e:
            print(f"[MLflow Warning] Could not log dataset: {e}")
    
    def log_dataset_details(self, folder_path, prefix="Train"):
        """Scans dataset, logs totals as parameters, and uploads file list as an artifact."""
        try:
            
            search_pattern = os.path.join(folder_path, "**", "*coord.npy")
            file_list = glob.glob(search_pattern, recursive=True)
            
            total_files = len(file_list)
            total_points = 0
            
            # Create a temporary text file to hold our beautiful list
            summary_filename = f"{prefix}_Dataset_Summary.txt"
            
            with open(summary_filename, "w") as f:
                f.write(f"=== {prefix.upper()} DATASET INFO ===\n")
                f.write(f"Path: {folder_path}\n")
                f.write(f"Total Files: {total_files}\n")
                f.write("-" * 40 + "\n")
                f.write("FILE LIST & POINT COUNTS:\n")
                
                # Loop through files, get their names, and count their points
                for path in file_list:
                    # 1. Step back one folder to get the true scan name (e.g., '56163_167883')
                    scan_name = os.path.basename(os.path.dirname(path))
                    
                    try:
                        # Load without putting it in RAM
                        data = np.load(path, mmap_mode='r')
                        points = data.shape[0] if isinstance(data, np.ndarray) else data.item()["coord"].shape[0]
                        total_points += points
                        
                        # 2. Write the scan name instead of "coord.npy"
                        f.write(f"{scan_name}: {points:,} points\n")
                        
                    except Exception as e:
                        f.write(f"{scan_name}: [Error reading points]\n")
                
                f.write("-" * 40 + "\n")
                f.write(f"GRAND TOTAL POINTS: {total_points:,}\n")
            
            # 1. Log the main numbers to the Overview dashboard
            mlflow.log_param(f"{prefix}_Files", total_files)
            mlflow.log_param(f"{prefix}_Points", f"{total_points:,}")
            
            # 2. Upload the text file so you can read the names in the Artifacts tab
            mlflow.log_artifact(summary_filename)
            
            # Clean up the text file from your local PC
            os.remove(summary_filename)
            
            print(f"[MLflow] Logged {prefix} details: {total_files} files, {total_points:,} points.")
            
        except Exception as e:
            print(f"[MLflow Warning] Could not log dataset details: {e}")

_global_mlflow_logger = None

def get_mlflow_logger():
    global _global_mlflow_logger
    return _global_mlflow_logger

def setup_mlflow_logger(tracking_uri, experiment_name, run_name):
    global _global_mlflow_logger
    _global_mlflow_logger = MLflowLogger(tracking_uri, experiment_name, run_name)
    return _global_mlflow_logger