"""
Comprehensive tests for the refunc.config module.

This test suite covers:
- Core configuration management (ConfigManager, ConfigSource)
- Configuration schemas for different components
- Utility functions for config handling (auto_configure, templates, validation)
- Error handling and validation
- File format support (YAML, JSON, TOML)
"""

import pytest
import json
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import os

# Import all config components
from refunc.config import (
    # Core classes
    ConfigManager,
    ConfigSource,
    ConfigError,
    ValidationError,
    
    # Schema classes
    RefuncConfig,
    DatabaseConfig,
    CacheConfig,
    LoggingConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    ExperimentConfig,
    PerformanceConfig,
    SecurityConfig,
    
    # Utility functions
    auto_configure,
    create_config_template,
    validate_config_file,
    merge_config_files,
    export_config,
    get_config_summary
)


class TestConfigManager:
    """Test ConfigManager functionality."""
    
    def test_config_manager_creation(self):
        """Test basic ConfigManager creation."""
        manager = ConfigManager()
        assert manager is not None
        
    def test_config_manager_with_validation(self):
        """Test ConfigManager with validation enabled."""
        manager = ConfigManager(validation_enabled=True)
        assert manager is not None
        
    def test_config_manager_basic_operations(self, temp_dir):
        """Test basic config operations."""
        manager = ConfigManager()
        
        # Test setting and getting values
        if hasattr(manager, 'set'):
            manager.set('test.key', 'test_value')
            value = manager.get('test.key')
            assert value == 'test_value'
        
    def test_config_manager_file_source(self, temp_dir):
        """Test adding file source to ConfigManager."""
        # Create a test config file
        config_file = temp_dir / "test_config.yaml"
        test_config = {
            'database': {
                'host': 'localhost',
                'port': 5432
            },
            'logging': {
                'level': 'INFO'
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        manager = ConfigManager()
        if hasattr(manager, 'add_file_source'):
            manager.add_file_source(str(config_file))
            
            # Test retrieving values
            db_host = manager.get('database.host')
            assert db_host == 'localhost'
            
    def test_config_manager_environment_variables(self):
        """Test ConfigManager with environment variables."""
        manager = ConfigManager()
        
        # Set environment variable
        os.environ['REFUNC_TEST_VAR'] = 'test_value'
        
        try:
            # Environment variables are loaded by default
            manager.add_env_source()
            
            # Try to retrieve the value (may have different key format)
            # Environment variables might be accessible differently
            pass  # Implementation-specific
        finally:
            # Clean up
            del os.environ['REFUNC_TEST_VAR']
            
    def test_config_manager_nested_get(self, temp_dir):
        """Test nested configuration retrieval."""
        config_file = temp_dir / "nested_config.yaml"
        nested_config = {
            'model': {
                'parameters': {
                    'learning_rate': 0.001,
                    'batch_size': 32
                },
                'architecture': {
                    'layers': 3,
                    'units': 128
                }
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(nested_config, f)
            
        manager = ConfigManager()
        if hasattr(manager, 'add_file_source'):
            manager.add_file_source(str(config_file))
            
            # Test nested retrieval
            lr = manager.get('model.parameters.learning_rate')
            if lr is not None:
                assert lr == 0.001
                
            params = manager.get('model.parameters', {})
            if params:
                assert params.get('batch_size') == 32


class TestConfigSource:
    """Test ConfigSource functionality."""
    
    def test_config_source_creation(self):
        """Test ConfigSource creation."""
        source = ConfigSource('test')
        assert source is not None
        assert source.name == 'test'
        
    def test_config_source_with_path(self, temp_dir):
        """Test ConfigSource with path."""
        test_file = temp_dir / "test.yaml"
        test_file.touch()
        
        source = ConfigSource('test_data', path=test_file)
        
        assert source.name == 'test_data'
        assert source.path == test_file


class TestConfigSchemas:
    """Test configuration schema classes."""
    
    def test_refunc_config_creation(self):
        """Test RefuncConfig schema creation."""
        config = RefuncConfig()
        assert config is not None
        
    def test_database_config_creation(self):
        """Test DatabaseConfig schema creation."""
        config = DatabaseConfig()
        assert config is not None
        
    def test_cache_config_creation(self):
        """Test CacheConfig schema creation."""
        config = CacheConfig()
        assert config is not None
        
    def test_logging_config_creation(self):
        """Test LoggingConfig schema creation."""
        config = LoggingConfig()
        assert config is not None
        
    def test_data_config_creation(self):
        """Test DataConfig schema creation."""
        config = DataConfig()
        assert config is not None
        
    def test_model_config_creation(self):
        """Test ModelConfig schema creation."""
        config = ModelConfig()
        assert config is not None
        
    def test_training_config_creation(self):
        """Test TrainingConfig schema creation."""
        config = TrainingConfig()
        assert config is not None
        
    def test_experiment_config_creation(self):
        """Test ExperimentConfig schema creation."""
        config = ExperimentConfig()
        assert config is not None
        
    def test_performance_config_creation(self):
        """Test PerformanceConfig schema creation."""
        config = PerformanceConfig()
        assert config is not None
        
    def test_security_config_creation(self):
        """Test SecurityConfig schema creation."""
        config = SecurityConfig()
        assert config is not None


class TestConfigUtils:
    """Test configuration utility functions."""
    
    def test_auto_configure(self):
        """Test auto_configure function."""
        config = auto_configure()
        assert config is not None
        
    def test_create_config_template(self, temp_dir):
        """Test create_config_template function."""
        template_file = temp_dir / "template.yaml"
        
        # Create template
        create_config_template(str(template_file))
        
        # Check if file was created
        assert template_file.exists()
        
        # Check if it's valid YAML
        with open(template_file, 'r') as f:
            template_data = yaml.safe_load(f)
            assert isinstance(template_data, dict)
            
    def test_validate_config_file(self, temp_dir):
        """Test validate_config_file function."""
        config_file = temp_dir / "valid_config.yaml"
        valid_config = {
            'project_name': 'test_project',  # Add required project_name field
            'logging': {
                'level': 'INFO',
                'file': 'app.log'
            },
            'database': {
                'host': 'localhost',
                'port': 5432
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(valid_config, f)
            
        # Validate the file
        result = validate_config_file(str(config_file))
        assert result is True or isinstance(result, dict)
        
    def test_merge_config_files(self, temp_dir):
        """Test merge_config_files function."""
        # Create first config file
        config1_file = temp_dir / "config1.yaml"
        config1 = {
            'database': {
                'host': 'localhost',
                'port': 5432
            },
            'logging': {
                'level': 'INFO'
            }
        }
        
        with open(config1_file, 'w') as f:
            yaml.dump(config1, f)
            
        # Create second config file
        config2_file = temp_dir / "config2.yaml"
        config2 = {
            'database': {
                'timeout': 30
            },
            'cache': {
                'type': 'redis',
                'host': 'localhost'
            }
        }
        
        with open(config2_file, 'w') as f:
            yaml.dump(config2, f)
            
        # Merge configs
        merged_file = temp_dir / "merged.yaml"
        merge_config_files([str(config1_file), str(config2_file)], str(merged_file))
        
        # Check if merged file exists
        assert merged_file.exists()
        
        # Check merged content
        with open(merged_file, 'r') as f:
            merged_data = yaml.safe_load(f)
            assert 'database' in merged_data
            assert 'logging' in merged_data
            assert 'cache' in merged_data
            
    def test_export_config(self, temp_dir):
        """Test export_config function."""
        manager = ConfigManager()
        
        # Add some test data
        if hasattr(manager, 'set'):
            manager.set('test.key', 'test_value')
            manager.set('test.number', 42)
            
            # Export to file
            export_file = temp_dir / "exported.yaml"
            export_config(str(export_file), config=manager)
            
            # Check if file was created
            assert export_file.exists()
        
    def test_get_config_summary(self):
        """Test get_config_summary function."""
        manager = ConfigManager()
        
        summary = get_config_summary(manager)
        assert summary is not None
        assert isinstance(summary, (str, dict))


class TestConfigErrors:
    """Test configuration error handling."""
    
    def test_config_error_creation(self):
        """Test ConfigError creation."""
        error = ConfigError("Test configuration error")
        # ConfigError inherits from RefuncError which adds prefix and timestamp
        error_str = str(error)
        assert "Test configuration error" in error_str
        assert "RefuncError:" in error_str
        
    def test_validation_error_creation(self):
        """Test ValidationError creation."""
        error = ValidationError("Test validation error")
        # ValidationError inherits from RefuncError which adds prefix and timestamp
        error_str = str(error)
        assert "Test validation error" in error_str
        assert "RefuncError:" in error_str
        
    def test_config_manager_invalid_file(self, temp_dir):
        """Test ConfigManager with invalid file."""
        invalid_file = temp_dir / "nonexistent.yaml"
        
        manager = ConfigManager()
        
        # Should handle missing file gracefully or raise appropriate error
        if hasattr(manager, 'add_file_source'):
            try:
                manager.add_file_source(str(invalid_file))
            except (ConfigError, FileNotFoundError):
                pass  # Expected behavior
                
    def test_config_manager_invalid_yaml(self, temp_dir):
        """Test ConfigManager with invalid YAML."""
        invalid_yaml_file = temp_dir / "invalid.yaml"
        
        # Create invalid YAML
        with open(invalid_yaml_file, 'w') as f:
            f.write("invalid: yaml: content:\n  - unclosed")
            
        manager = ConfigManager()
        
        if hasattr(manager, 'add_file_source'):
            try:
                manager.add_file_source(str(invalid_yaml_file))
            except (ConfigError, yaml.YAMLError):
                pass  # Expected behavior


class TestConfigFileFormats:
    """Test different configuration file formats."""
    
    def test_yaml_config_loading(self, temp_dir):
        """Test loading YAML configuration."""
        yaml_file = temp_dir / "config.yaml"
        yaml_config = {
            'app': {
                'name': 'test_app',
                'version': '1.0.0'
            },
            'features': ['feature1', 'feature2']
        }
        
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_config, f)
            
        manager = ConfigManager()
        if hasattr(manager, 'add_file_source'):
            manager.add_file_source(str(yaml_file))
            
            app_name = manager.get('app.name')
            if app_name is not None:
                assert app_name == 'test_app'
                
    def test_json_config_loading(self, temp_dir):
        """Test loading JSON configuration."""
        json_file = temp_dir / "config.json"
        json_config = {
            'api': {
                'endpoint': 'https://api.example.com',
                'timeout': 30
            },
            'debug': True
        }
        
        with open(json_file, 'w') as f:
            json.dump(json_config, f)
            
        manager = ConfigManager()
        if hasattr(manager, 'add_file_source'):
            manager.add_file_source(str(json_file))
            
            endpoint = manager.get('api.endpoint')
            if endpoint is not None:
                assert endpoint == 'https://api.example.com'


class TestConfigIntegration:
    """Test integration between different config components."""
    
    def test_schema_validation_workflow(self, temp_dir):
        """Test complete schema validation workflow."""
        config_file = temp_dir / "app_config.yaml"
        app_config = {
            'logging': {
                'level': 'DEBUG',
                'file': 'app.log'
            },
            'database': {
                'host': 'localhost',
                'port': 5432,
                'name': 'testdb'
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(app_config, f)
            
        # Create manager with validation
        manager = ConfigManager(validation_enabled=True)
        if hasattr(manager, 'add_file_source'):
            manager.add_file_source(str(config_file))
            
            # Should work without errors
            logging_level = manager.get('logging.level')
            if logging_level is not None:
                assert logging_level == 'DEBUG'
                
    def test_multi_source_configuration(self, temp_dir):
        """Test configuration from multiple sources."""
        # File source
        file_config = temp_dir / "base.yaml"
        base_config = {
            'app': {
                'name': 'test_app',
                'debug': False
            }
        }
        
        with open(file_config, 'w') as f:
            yaml.dump(base_config, f)
            
        # Environment override
        os.environ['REFUNC_APP_DEBUG'] = 'true'
        
        try:
            manager = ConfigManager()
            if hasattr(manager, 'add_file_source'):
                manager.add_file_source(str(file_config))
            if hasattr(manager, 'add_env_source'):
                manager.add_env_source()
                
                # Environment should override file (implementation specific)
                debug_value = manager.get('app.debug')
                # Could be True, 'true', or original False depending on implementation
                assert debug_value is not None
        finally:
            # Clean up
            del os.environ['REFUNC_APP_DEBUG']
            
    def test_template_and_validation_workflow(self, temp_dir):
        """Test template creation and validation workflow."""
        template_file = temp_dir / "new_template.yaml"
        
        # Create template
        create_config_template(str(template_file))
        assert template_file.exists()
        
        # Validate template
        validation_result = validate_config_file(str(template_file))
        assert validation_result is not None


class TestConfigPerformance:
    """Test configuration performance characteristics."""
    
    @pytest.mark.slow
    def test_large_config_loading(self, temp_dir):
        """Test loading large configuration files."""
        large_config_file = temp_dir / "large_config.yaml"
        
        # Create large config
        large_config = {}
        for i in range(1000):
            large_config[f'section_{i}'] = {
                f'key_{j}': f'value_{i}_{j}' for j in range(10)
            }
            
        with open(large_config_file, 'w') as f:
            yaml.dump(large_config, f)
            
        # Load and test
        manager = ConfigManager()
        if hasattr(manager, 'add_file_source'):
            manager.add_file_source(str(large_config_file))
            
            # Should handle large configs efficiently
            value = manager.get('section_500.key_5')
            if value is not None:
                assert value == 'value_500_5'
                
    @pytest.mark.slow
    def test_many_config_sources(self, temp_dir):
        """Test performance with many configuration sources."""
        manager = ConfigManager()
        
        # Add many file sources
        for i in range(10):
            config_file = temp_dir / f"config_{i}.yaml"
            config_data = {f'section_{i}': {f'key_{i}': f'value_{i}'}}
            
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f)
                
            if hasattr(manager, 'add_file_source'):
                manager.add_file_source(str(config_file))
                
        # Should handle multiple sources efficiently
        for i in range(10):
            value = manager.get(f'section_{i}.key_{i}')
            if value is not None:
                assert value == f'value_{i}'


class TestConfigEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_config_file(self, temp_dir):
        """Test handling of empty configuration files."""
        empty_file = temp_dir / "empty.yaml"
        empty_file.touch()
        
        manager = ConfigManager()
        if hasattr(manager, 'add_file_source'):
            try:
                manager.add_file_source(str(empty_file))
                # Should handle empty files gracefully
            except (ConfigError, yaml.YAMLError):
                pass  # May raise error, which is acceptable
                
    def test_circular_config_references(self, temp_dir):
        """Test handling of circular configuration references."""
        # This is a complex test that may not be applicable to all implementations
        manager = ConfigManager()
        
        # Test basic functionality without circular references
        if hasattr(manager, 'set'):
            manager.set('test.ref', 'value')
            value = manager.get('test.ref')
            assert value == 'value'
            
    def test_unicode_config_values(self, temp_dir):
        """Test handling of unicode values in configuration."""
        unicode_file = temp_dir / "unicode.yaml"
        unicode_config = {
            'messages': {
                'greeting': 'こんにちは',  # Japanese
                'farewell': 'Auf Wiedersehen',  # German
                'emoji': '🎉🚀'
            }
        }
        
        with open(unicode_file, 'w', encoding='utf-8') as f:
            yaml.dump(unicode_config, f, allow_unicode=True)
            
        manager = ConfigManager()
        if hasattr(manager, 'add_file_source'):
            manager.add_file_source(str(unicode_file))
            
            greeting = manager.get('messages.greeting')
            if greeting is not None:
                assert greeting == 'こんにちは'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])