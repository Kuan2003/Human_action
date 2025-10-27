#!/usr/bin/env python3
"""
Environment Configuration for Human Action Recognition Project
Supports multiple deployment scenarios
"""

import os
import json
from pathlib import Path

class EnvironmentConfig:
    """
    Handles environment-specific configurations
    Supports: development, server, docker, colab, etc.
    """
    
    def __init__(self, config_file=None):
        self.config_file = config_file
        self.config = self._load_config()
        self._setup_environment()
    
    def _load_config(self):
        """Load configuration from multiple sources"""
        config = {
            # Default configuration
            'environment': 'auto',  # auto, development, server, docker, colab
            'project_name': 'Human_action',
            'data_dir': 'processed_data_auto',
            'models_dir': 'outputs',
            'logs_dir': 'logs',
            'cache_dir': '.cache',
            
            # Path resolution settings
            'path_resolution': {
                'strict_mode': False,  # If True, fail if files don't exist
                'cache_enabled': True,
                'auto_create_dirs': True
            },
            
            # Training settings
            'training': {
                'device': 'auto',  # auto, cpu, cuda, mps
                'num_workers': 4,
                'pin_memory': True
            },
            
            # Environment-specific overrides
            'environment_overrides': {}
        }
        
        # Load from config file if provided
        if self.config_file and os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                user_config = json.load(f)
                config.update(user_config)
        
        # Load from environment variables
        env_overrides = {}
        for key in os.environ:
            if key.startswith('HAR_'):  # Human Action Recognition prefix
                config_key = key[4:].lower()  # Remove HAR_ prefix
                env_overrides[config_key] = os.environ[key]
        
        if env_overrides:
            config['environment_overrides'].update(env_overrides)
        
        return config
    
    def _setup_environment(self):
        """Setup environment based on detection and config"""
        
        # Auto-detect environment if not specified
        if self.config['environment'] == 'auto':
            self.config['environment'] = self._detect_environment()
        
        # Apply environment-specific settings
        env = self.config['environment']
        
        if env == 'colab':
            self._setup_colab()
        elif env == 'docker':
            self._setup_docker()
        elif env == 'server':
            self._setup_server()
        elif env == 'development':
            self._setup_development()
        
        # Apply overrides
        for key, value in self.config['environment_overrides'].items():
            self._set_nested_config(key, value)
    
    def _detect_environment(self):
        """Auto-detect current environment"""
        
        # Google Colab
        if 'COLAB_GPU' in os.environ or 'google.colab' in str(os.environ.get('JPY_PARENT_PID', '')):
            return 'colab'
        
        # Docker container
        if os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER'):
            return 'docker'
        
        # Server/cluster (common patterns)
        server_indicators = [
            '/home/ubuntu/',  # AWS/cloud
            '/data/',         # Common data mount
            '/scratch/',      # HPC scratch space
            'SLURM_JOB_ID',   # SLURM cluster
            'PBS_JOBID',      # PBS cluster
        ]
        
        cwd = os.getcwd()
        if (any(indicator in cwd for indicator in server_indicators[:3]) or
            any(indicator in os.environ for indicator in server_indicators[3:])):
            return 'server'
        
        # Development (default)
        return 'development'
    
    def _setup_colab(self):
        """Setup for Google Colab"""
        self.config['training']['num_workers'] = 2  # Colab limitation
        self.config['path_resolution']['auto_create_dirs'] = True
        
        # Common Colab mount points
        if os.path.exists('/content/drive'):
            self.config['colab_drive'] = '/content/drive/MyDrive'
    
    def _setup_docker(self):
        """Setup for Docker container"""
        self.config['training']['num_workers'] = min(4, os.cpu_count() or 4)
        
        # Common Docker volume mounts
        docker_mounts = ['/app', '/workspace', '/data']
        for mount in docker_mounts:
            if os.path.exists(mount):
                self.config['docker_workspace'] = mount
                break
    
    def _setup_server(self):
        """Setup for server/cluster environment"""
        self.config['training']['num_workers'] = min(8, os.cpu_count() or 4)
        self.config['path_resolution']['strict_mode'] = True
        
        # Use environment modules if available
        if 'MODULEPATH' in os.environ:
            self.config['uses_modules'] = True
    
    def _setup_development(self):
        """Setup for development environment"""
        self.config['training']['num_workers'] = min(4, os.cpu_count() or 4)
        self.config['path_resolution']['cache_enabled'] = True
    
    def _set_nested_config(self, key, value):
        """Set nested configuration value using dot notation"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def get(self, key, default=None):
        """Get configuration value using dot notation"""
        keys = key.split('.')
        config = self.config
        for k in keys:
            if isinstance(config, dict) and k in config:
                config = config[k]
            else:
                return default
        return config
    
    def get_device(self):
        """Get appropriate device for training"""
        device_setting = self.get('training.device', 'auto')
        
        if device_setting == 'auto':
            if torch_available := self._is_torch_available():
                import torch
                if torch.cuda.is_available():
                    return 'cuda'
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return 'mps'
                else:
                    return 'cpu'
            else:
                return 'cpu'
        
        return device_setting
    
    def _is_torch_available(self):
        """Check if PyTorch is available"""
        try:
            import torch
            return True
        except ImportError:
            return False
    
    def create_directories(self, project_root):
        """Create necessary directories"""
        if not self.get('path_resolution.auto_create_dirs', True):
            return
        
        dirs_to_create = [
            self.get('models_dir', 'outputs'),
            self.get('logs_dir', 'logs'),
            self.get('cache_dir', '.cache'),
        ]
        
        for dir_name in dirs_to_create:
            dir_path = os.path.join(project_root, dir_name)
            os.makedirs(dir_path, exist_ok=True)
    
    def save_config(self, file_path):
        """Save current configuration to file"""
        with open(file_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def print_summary(self):
        """Print configuration summary"""
        print("="*60)
        print("ENVIRONMENT CONFIGURATION")
        print("="*60)
        print(f"Environment: {self.config['environment']}")
        print(f"Device: {self.get_device()}")
        print(f"Workers: {self.get('training.num_workers')}")
        print(f"Strict mode: {self.get('path_resolution.strict_mode')}")
        print(f"Auto create dirs: {self.get('path_resolution.auto_create_dirs')}")
        
        if self.config['environment'] == 'colab' and 'colab_drive' in self.config:
            print(f"Colab drive: {self.config['colab_drive']}")
        
        if self.config['environment'] == 'docker' and 'docker_workspace' in self.config:
            print(f"Docker workspace: {self.config['docker_workspace']}")
        
        print("="*60)

# Global configuration instance
_global_config = None

def get_config(config_file=None):
    """Get global configuration instance"""
    global _global_config
    if _global_config is None or config_file:
        _global_config = EnvironmentConfig(config_file)
    return _global_config

def setup_environment(config_file=None, project_root=None):
    """
    Quick environment setup
    
    Usage:
        from environment_config import setup_environment
        config = setup_environment()
        device = config.get_device()
    """
    config = get_config(config_file)
    
    if project_root:
        config.create_directories(project_root)
    
    return config

# Example usage
if __name__ == "__main__":
    config = EnvironmentConfig()
    config.print_summary()
    
    # Test configuration access
    print(f"\nDevice: {config.get_device()}")
    print(f"Data directory: {config.get('data_dir')}")
    print(f"Models directory: {config.get('models_dir')}")
    print(f"Workers: {config.get('training.num_workers')}")
    
    # Save example config
    config.save_config('example_config.json')
    print("\nExample configuration saved to 'example_config.json'")