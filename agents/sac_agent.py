import numpy as np
import yaml
import json
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import tempfile
import os

from environments.stadium_track import StadiumTrackEnv
from agents.optimal_control_agent import OptimalControlAgent


class ExperimentRunner:
    """
    Run final experiments with best hyperparameters found from tuning.
    """
    
    def __init__(self, config_path: str = "config.yaml", results_path: str = "hyperparameter_results.json"):
        with open(config_path) as f:
            self.base_config = yaml.safe_load(f)
        
        # Load best hyperparameters if available
        self.best_hyperparams = {}
        if os.path.exists(results_path):
            with open(results_path) as f:
                tuning_results = json.load(f)
                for agent_type in ['oc', 'ppo', 'sac']:
                    if agent_type in tuning_results.get('analysis', {}):
                        self.best_hyperparams[agent_type] = tuning_results['analysis'][agent_type]['best_hyperparams']
        
        # Default hyperparameters if tuning results not available
        if not self.best_hyperparams:
            self.best_hyperparams = {
                'oc': {
                    'turning_radius': 10.0,
                    'max_steering_rate_abs': np.pi / 4,
                    'max_acceleration': 2.0,
                    'min_acceleration': -3.0
                },
                'ppo': {
                    'learning_rate': 3e-4,
                    'n_steps': 2048,
                    'batch_size': 64,
                    'gamma': 0.99,
                    'gae_lambda': 0.95,
                    'clip_range': 0.2,
                    'ent_coef': 0.01,
                    'vf_coef': 0.5,
                    'max_grad_norm': 0.5
                },
                'sac': {
                    'learning_rate': 3e-4,
                    'buffer_size': 100000,
                    'learning_starts': 10000,
                    'batch_size': 128,
                    'tau': 0.005,
                    'gamma': 0.99,
                    'train_freq': 4,
                    'gradient_steps': 1,
                    'ent_coef': 'auto',
                    'target_update_interval': 1
                }
            }
    
    def create_env(self, seed: int = None) -> Monitor:
        """Create monitored environment."""
        env = StadiumTrackEnv(self.base_config['env'])
        env = Monitor(env)
        if seed is not None:
            env.seed(seed)
        return env
    
    def train_and_evaluate_ppo(self, total_timesteps: int = 200000, seed: int = 42) -> dict:
        """Train PPO with best hyperparameters and evaluate."""
        print("Training PPO agent with best hyperparameters...")
        
        env = self.create_env(seed)
        eval_env = self.create_env(seed + 1)
        
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=seed,
            **self.best_hyperparams['ppo']
        )
        
        # Setup evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="./best_ppo_model",
            log_path="./ppo_logs",
            eval_freq=10000,
            n_eval_episodes=10,
            deterministic=True
        )
        
        # Train model
        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        
        # Load best model
        best_model_path = "./best_ppo_model/best_model.zip"
        if os.path.exists(best_model_path):
            model = PPO.load(best_model_path)
        
        # Save final model
        model.save("final_ppo_model")
        
        # Evaluate final performance
        return self.evaluate_agent(model, eval_env, num_episodes=50)
    
    def train_and_evaluate_sac(self, total_timesteps: int = 200000, seed: int = 42) -> dict:
        """Train SAC with best hyperparameters and evaluate."""
        print("Training SAC agent with best hyperparameters...")
        
        env = self.create_env(seed)
        eval_env = self.create_env(seed + 1)
        
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            seed=seed,
            **self.best_hyperparams['sac']
        )
        
        # Setup evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="./best_sac_model",
            log_path="./sac_logs",
            eval_freq=10000,
            n_eval_episodes=10,
            deterministic=True
        )
        
        # Train model
        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        
        # Load best model
        best_model_path = "./best_sac_model/best_model.zip"
        if os.path.exists(best_model_path):
            model = SAC.load(best_model_path)
        
        # Save final model
        model.save("final_sac_model")
        
        # Evaluate final performance
        return self.evaluate_
