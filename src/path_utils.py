#!/usr/bin/env python3
"""
Universal Path Resolution Utilities
Handles file paths across different environments and deployment scenarios
"""

import os
import sys
from pathlib import Path

class SmartPathResolver:
    """
    Intelligent path resolver that works across different environments:
    - Development machine (your laptop)
    - Different servers/clusters  
    - Docker containers
    - Different user accounts
    - Different operating systems
    """
    
    def __init__(self, anchor_file=None):
        """
        Initialize with an anchor file (like dataset.json) to find project root
        """
        self.anchor_file = anchor_file
        self.project_root = self._find_project_root()
        self.cache = {}  # Cache resolved paths for performance
        
    def _find_project_root(self):
        """
        Find project root using multiple strategies
        """
        # Strategy 1: Use anchor file location
        if self.anchor_file and os.path.exists(self.anchor_file):
            anchor_dir = os.path.dirname(os.path.abspath(self.anchor_file))
            root = self._traverse_up_for_markers(anchor_dir)
            if root:
                return root
        
        # Strategy 2: Use current script location
        current_script = os.path.abspath(__file__)
        script_dir = os.path.dirname(current_script)
        root = self._traverse_up_for_markers(script_dir)
        if root:
            return root
            
        # Strategy 3: Use current working directory
        cwd = os.getcwd()
        root = self._traverse_up_for_markers(cwd)
        if root:
            return root
            
        # Strategy 4: Environment variable
        if 'PROJECT_ROOT' in os.environ:
            env_root = os.environ['PROJECT_ROOT']
            if os.path.exists(env_root):
                return env_root
                
        # Strategy 5: Default fallback
        return os.getcwd()
    
    def _traverse_up_for_markers(self, start_dir, max_levels=8):
        """
        Traverse up directory tree looking for project markers
        """
        project_markers = [
            'processed_data_auto',   # Our main data folder
            'recorded_dataset',      # Raw data
            'Human_action',         # Project folder
            '.git',                 # Git repo
            'requirements.txt',     # Python project
            'setup.py',            # Python package
            'pyproject.toml',      # Modern Python project
            'README.md',           # Documentation
            'train_stageA.py',     # Our specific files
            'Pipfile',             # Pipenv
            'environment.yml'      # Conda environment
        ]
        
        current = os.path.abspath(start_dir)
        
        for _ in range(max_levels):
            try:
                contents = os.listdir(current)
                
                # Strong indicators (definitely project root)
                strong_markers = ['processed_data_auto', 'Human_action', '.git']
                if any(marker in contents for marker in strong_markers):
                    # Special case: if we're inside processed_data_auto, go up
                    if os.path.basename(current) == 'processed_data_auto':
                        return os.path.dirname(current)
                    # If processed_data_auto is in current dir, this is root
                    if 'processed_data_auto' in contents:
                        return current
                    # If Human_action is here, this might be parent of project
                    if 'Human_action' in contents:
                        return current
                
                # Weak indicators (might be project root)
                weak_markers = ['requirements.txt', 'README.md', 'setup.py']
                if any(marker in contents for marker in weak_markers):
                    # Additional validation - check for our specific structure
                    if any(os.path.exists(os.path.join(current, subdir)) 
                           for subdir in ['src', 'Source', 'Human_action']):
                        return current
                
                parent = os.path.dirname(current)
                if parent == current:  # Reached filesystem root
                    break
                current = parent
                
            except (OSError, PermissionError):
                break
                
        return None
    
    def resolve(self, file_path, must_exist=True):
        """
        Resolve file path using multiple strategies
        
        Args:
            file_path: Original file path (can be relative or absolute)
            must_exist: If True, only return paths that exist
            
        Returns:
            Resolved absolute path, or original path if resolution fails
        """
        if not file_path:
            return file_path
            
        # Check cache first
        cache_key = (file_path, must_exist)
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        resolved = self._resolve_internal(file_path, must_exist)
        self.cache[cache_key] = resolved
        return resolved
    
    def _resolve_internal(self, file_path, must_exist):
        """Internal resolution logic"""
        
        # Strategy 1: Already absolute and exists
        if os.path.isabs(file_path):
            if not must_exist or os.path.exists(file_path):
                return file_path
        
        # Strategy 2: Relative to project root
        candidate = os.path.join(self.project_root, file_path)
        if not must_exist or os.path.exists(candidate):
            return os.path.abspath(candidate)
            
        # Strategy 3: Relative to anchor file directory
        if self.anchor_file:
            anchor_dir = os.path.dirname(self.anchor_file)
            candidate = os.path.join(anchor_dir, file_path)
            if not must_exist or os.path.exists(candidate):
                return os.path.abspath(candidate)
                
        # Strategy 4: Relative to current working directory
        candidate = os.path.join(os.getcwd(), file_path)
        if not must_exist or os.path.exists(candidate):
            return os.path.abspath(candidate)
            
        # Strategy 5: Try common subdirectories
        common_subdirs = [
            '',
            'processed_data_auto',
            'Human_action',
            'Human_action/Source',
            'Source',
            'data',
            'datasets'
        ]
        
        for subdir in common_subdirs:
            candidate = os.path.join(self.project_root, subdir, file_path)
            if not must_exist or os.path.exists(candidate):
                return os.path.abspath(candidate)
                
        # Strategy 6: Handle processed_data_auto prefix
        if file_path.startswith('processed_data_auto/'):
            rel_path = file_path[len('processed_data_auto/'):]
            processed_dir = os.path.join(self.project_root, 'processed_data_auto')
            candidate = os.path.join(processed_dir, rel_path)
            if not must_exist or os.path.exists(candidate):
                return os.path.abspath(candidate)
                
        # Strategy 7: Try removing common prefixes
        for prefix in ['data/', 'datasets/', 'frames/']:
            if file_path.startswith(prefix):
                rel_path = file_path[len(prefix):]
                candidate = os.path.join(self.project_root, 'processed_data_auto', rel_path)
                if not must_exist or os.path.exists(candidate):
                    return os.path.abspath(candidate)
        
        # Strategy 8: Last resort - return original path
        return file_path
    
    def get_project_root(self):
        """Get the detected project root directory"""
        return self.project_root
    
    def validate_paths(self, paths):
        """
        Validate multiple paths and return resolution report
        
        Args:
            paths: List of file paths or dict of {name: path}
            
        Returns:
            Dict with resolution results
        """
        if isinstance(paths, dict):
            items = paths.items()
        else:
            items = enumerate(paths)
            
        results = {
            'resolved': {},
            'missing': {},
            'total': len(list(items)),
            'success_rate': 0
        }
        
        success_count = 0
        for name, path in items:
            resolved = self.resolve(path, must_exist=True)
            if os.path.exists(resolved):
                results['resolved'][name] = resolved
                success_count += 1
            else:
                results['missing'][name] = path
                
        results['success_rate'] = success_count / max(1, results['total'])
        return results

# Global instance for easy usage
_global_resolver = None

def get_resolver(anchor_file=None):
    """Get global path resolver instance"""
    global _global_resolver
    if _global_resolver is None or anchor_file:
        _global_resolver = SmartPathResolver(anchor_file)
    return _global_resolver

def resolve_path(file_path, must_exist=True, anchor_file=None):
    """
    Quick path resolution function
    
    Usage:
        resolved_path = resolve_path('processed_data_auto/frames/rgb_000001.png')
        resolved_path = resolve_path('/abs/path/file.txt')
        resolved_path = resolve_path('relative/path.json', anchor_file='dataset.json')
    """
    resolver = get_resolver(anchor_file)
    return resolver.resolve(file_path, must_exist)

def get_project_root(anchor_file=None):
    """Get project root directory"""
    resolver = get_resolver(anchor_file)
    return resolver.get_project_root()

def set_project_root(root_path):
    """Manually set project root (useful for deployment)"""
    os.environ['PROJECT_ROOT'] = root_path
    global _global_resolver
    _global_resolver = None  # Force recreation

# Example usage and testing
if __name__ == "__main__":
    resolver = SmartPathResolver()
    print(f"Project root: {resolver.get_project_root()}")
    
    # Test some paths
    test_paths = [
        'processed_data_auto/dataset.json',
        'processed_data_auto/frames/rgb_000001.png',
        'Human_action/Source/src/models.py',
        '/absolute/path/that/might/not/exist.txt'
    ]
    
    for path in test_paths:
        resolved = resolver.resolve(path, must_exist=False)
        exists = os.path.exists(resolved)
        print(f"'{path}' -> '{resolved}' (exists: {exists})")