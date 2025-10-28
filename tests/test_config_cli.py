import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

from refunc.config.cli import (
    create_parser,
    cmd_template,
    cmd_validate,
    cmd_merge,
    cmd_show,
    cmd_export,
    main
)


class TestCLIBasic:
    """Test basic CLI functionality."""
    
    def test_create_parser(self):
        """Test parser creation and basic structure."""
        parser = create_parser()
        assert parser is not None
        
        # Test help option
        with pytest.raises(SystemExit):
            parser.parse_args(['--help'])
    
    def test_parser_template_command(self):
        """Test template command parsing."""
        parser = create_parser()
        
        # Template command - output is positional argument
        args = parser.parse_args(['template', 'test.yaml'])
        assert args.command == 'template'
        assert args.output == 'test.yaml'
        assert args.type == 'full'  # default
        assert args.format == 'yaml'  # default
    
    def test_parser_validate_command(self):
        """Test validate command parsing."""
        parser = create_parser()
        
        # Validate command
        args = parser.parse_args(['validate', 'config.yaml'])
        assert args.command == 'validate'
        assert args.config_file == 'config.yaml'
        assert args.schema == 'refunc'  # default
    
    def test_parser_merge_command(self):
        """Test merge command parsing."""
        parser = create_parser()
        
        # Merge command
        args = parser.parse_args(['merge', 'file1.yaml', 'file2.yaml', '--output', 'merged.yaml'])
        assert args.command == 'merge'
        assert args.input_files == ['file1.yaml', 'file2.yaml']
        assert args.output == 'merged.yaml'
    
    def test_parser_show_command(self):
        """Test show command parsing."""
        parser = create_parser()
        
        # Show command
        args = parser.parse_args(['show'])
        assert args.command == 'show'
        assert args.config_files is None  # default
        assert args.format == 'summary'  # default
    
    def test_parser_export_command(self):
        """Test export command parsing."""
        parser = create_parser()
        
        # Export command - output is positional argument
        args = parser.parse_args(['export', 'export.yaml'])
        assert args.command == 'export'
        assert args.output == 'export.yaml'


class TestCLICommands:
    """Test individual CLI commands."""
    
    def test_cmd_template_success(self, capsys):
        """Test template command with successful file creation."""
        temp_dir = tempfile.mkdtemp()
        try:
            temp_file = os.path.join(temp_dir, 'template.yaml')
            
            args = Mock()
            args.output = temp_file
            args.type = 'basic'
            args.format = 'yaml'
            args.no_comments = False
            
            cmd_template(args)
            
            captured = capsys.readouterr()
            assert 'Template created successfully' in captured.out
            assert os.path.exists(temp_file)
            
        finally:
            # Cleanup
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            os.rmdir(temp_dir)
    
    def test_cmd_validate_missing_file(self, capsys):
        """Test validate command with missing file."""
        args = Mock()
        args.config_file = 'nonexistent.yaml'
        args.schema = 'refunc'
        
        with pytest.raises(SystemExit) as exc_info:
            cmd_validate(args)
        
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert 'Configuration file is invalid' in captured.out
    
    def test_cmd_merge_missing_files(self, capsys):
        """Test merge command with missing files."""
        args = Mock()
        args.input_files = ['missing1.yaml', 'missing2.yaml']
        args.output = 'merged.yaml'
        args.format = 'yaml'
        
        with pytest.raises(SystemExit) as exc_info:
            cmd_merge(args)
        
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert 'Error merging configurations' in captured.out
    
    def test_cmd_show_with_key(self, capsys):
        """Test show command with specific key."""
        args = Mock()
        args.config_files = None
        args.format = 'summary'
        args.key = 'nonexistent'
        args.no_metadata = False
        
        with pytest.raises(SystemExit) as exc_info:
            cmd_show(args)
        
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert 'not found' in captured.out
    
    def test_cmd_export_success(self, capsys):
        """Test export command basic functionality."""
        temp_dir = tempfile.mkdtemp()
        try:
            temp_file = os.path.join(temp_dir, 'export.yaml')
            
            args = Mock()
            args.output = temp_file
            args.config_files = None
            args.format = 'yaml'
            args.no_metadata = False
            
            cmd_export(args)
            
            captured = capsys.readouterr()
            assert 'Configuration exported successfully' in captured.out
            
        finally:
            # Cleanup
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            os.rmdir(temp_dir)


class TestCLIMain:
    """Test main function and argument handling."""
    
    @patch('refunc.config.cli.create_parser')
    def test_main_template_command(self, mock_create_parser):
        """Test main function dispatches template command."""
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.command = 'template'
        mock_parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = mock_parser
        
        with patch('refunc.config.cli.cmd_template') as mock_cmd:
            main()
            mock_cmd.assert_called_once_with(mock_args)
    
    @patch('refunc.config.cli.create_parser')
    def test_main_validate_command(self, mock_create_parser):
        """Test main function dispatches validate command."""
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.command = 'validate'
        mock_parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = mock_parser
        
        with patch('refunc.config.cli.cmd_validate') as mock_cmd:
            main()
            mock_cmd.assert_called_once_with(mock_args)
    
    @patch('refunc.config.cli.create_parser')
    def test_main_merge_command(self, mock_create_parser):
        """Test main function dispatches merge command."""
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.command = 'merge'
        mock_parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = mock_parser
        
        with patch('refunc.config.cli.cmd_merge') as mock_cmd:
            main()
            mock_cmd.assert_called_once_with(mock_args)
    
    @patch('refunc.config.cli.create_parser')
    def test_main_show_command(self, mock_create_parser):
        """Test main function dispatches show command."""
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.command = 'show'
        mock_parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = mock_parser
        
        with patch('refunc.config.cli.cmd_show') as mock_cmd:
            main()
            mock_cmd.assert_called_once_with(mock_args)
    
    @patch('refunc.config.cli.create_parser')
    def test_main_export_command(self, mock_create_parser):
        """Test main function dispatches export command."""
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.command = 'export'
        mock_parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = mock_parser
        
        with patch('refunc.config.cli.cmd_export') as mock_cmd:
            main()
            mock_cmd.assert_called_once_with(mock_args)
    
    @patch('refunc.config.cli.create_parser')
    def test_main_no_command(self, mock_create_parser):
        """Test main function when no command provided."""
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.command = None
        mock_parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = mock_parser
        
        # Should handle gracefully when no command provided
        main()
        # No assertion needed - just ensure it doesn't crash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
