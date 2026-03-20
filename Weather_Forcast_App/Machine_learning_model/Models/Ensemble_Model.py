"""Backward-compat shim so old pickles that reference Ensemble_Model still load."""
from Weather_Forcast_App.Machine_learning_model.Models.Ensemble_Average_Model import *  # noqa: F401,F403
from Weather_Forcast_App.Machine_learning_model.Models.Ensemble_Average_Model import WeatherEnsembleModel  # noqa: F401
