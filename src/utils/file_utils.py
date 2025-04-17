"""
Utility functions for file and directory operations.
"""
import os
import logging


def get_project_root():
    """
    Get the absolute path to the project root directory.
    Returns:
        str: The absolute path to the project root directory.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def create_run_folder(base_path="outputs"):
    """
    Create a new numbered run folder (e.g., run-1, run-2) with a subfolder for plots.
    Args:
        base_path (str): The base path where the run folder will be created.
    Returns:
        run_path (str): The path to the created run folder. 
    """

    try:
        # Project absolute path to the base directory
        project_root = get_project_root()

        # Join the project root with the base path
        full_base_path = os.path.join(project_root, base_path)

        run_number = 1

        # Check for existing run folders and find the next available run number
        while os.path.exists(os.path.join(full_base_path, f"run-{run_number}")):
            run_number += 1

        # Create a new run folder with the current run number
        run_path = os.path.join(full_base_path, f"run-{run_number}")

        # Create a subfolder for plots
        plots_path = os.path.join(run_path, "plots")

        # Create the directories if they do not exist
        os.makedirs(run_path, exist_ok=True)
        os.makedirs(plots_path, exist_ok=True)

    except Exception as e:
        print(f"Error creating folders: {e}")
        raise
    
    return run_path

def configure_logger(run_path, log_name="run.log"):
    """
    Configure a logger to log messages to both a file and the console.
    Args:
        run_path (str): The path where the log file will be created.
        log_name (str): The name of the log file.
    Returns:
        logger (logging.Logger): The configured logger instance.
    """
    # Create a logger
    logger = logging.getLogger("lstm_logger")
    logger.setLevel(logging.INFO)

    # Check if the logger already has handlers to avoid adding multiple handlers
    if not logger.handlers:
        log_path = os.path.join(run_path, log_name)

        # Create the format for the log messages
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        # File handler
        fh = logging.FileHandler(log_path, mode='a')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # To show logs in the console as well
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger
