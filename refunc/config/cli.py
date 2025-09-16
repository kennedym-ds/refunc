"""
Command-line interface for configuration management.

This module provides CLI commands for managing refunc configurations,
including template generation, validation, and merging.
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional

from .core import ConfigManager
from .schemas import RefuncConfig
from .utils import (
    create_config_template,
    validate_config_file,
    merge_config_files,
    auto_configure,
    export_config,
    get_config_summary
)


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="refunc-config",
        description="Refunc configuration management CLI"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Template command
    template_parser = subparsers.add_parser(
        "template",
        help="Generate configuration template"
    )
    template_parser.add_argument(
        "output",
        help="Output file path"
    )
    template_parser.add_argument(
        "--type",
        choices=["full", "development", "production", "training", "inference"],
        default="full",
        help="Template type"
    )
    template_parser.add_argument(
        "--format",
        choices=["yaml", "json"],
        default="yaml",
        help="Output format"
    )
    template_parser.add_argument(
        "--no-comments",
        action="store_true",
        help="Exclude comments from template"
    )
    
    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate configuration file"
    )
    validate_parser.add_argument(
        "config_file",
        help="Configuration file to validate"
    )
    validate_parser.add_argument(
        "--schema",
        choices=["refunc", "training", "inference"],
        default="refunc",
        help="Schema to validate against"
    )
    
    # Merge command
    merge_parser = subparsers.add_parser(
        "merge",
        help="Merge multiple configuration files"
    )
    merge_parser.add_argument(
        "input_files",
        nargs="+",
        help="Input configuration files"
    )
    merge_parser.add_argument(
        "--output",
        required=True,
        help="Output file path"
    )
    merge_parser.add_argument(
        "--format",
        choices=["yaml", "json"],
        default="yaml",
        help="Output format"
    )
    
    # Show command
    show_parser = subparsers.add_parser(
        "show",
        help="Show current configuration"
    )
    show_parser.add_argument(
        "--config-files",
        nargs="+",
        help="Configuration files to load"
    )
    show_parser.add_argument(
        "--format",
        choices=["yaml", "json", "summary"],
        default="summary",
        help="Output format"
    )
    show_parser.add_argument(
        "--key",
        help="Show specific configuration key"
    )
    
    # Export command
    export_parser = subparsers.add_parser(
        "export",
        help="Export current configuration"
    )
    export_parser.add_argument(
        "output",
        help="Output file path"
    )
    export_parser.add_argument(
        "--config-files",
        nargs="+",
        help="Configuration files to load"
    )
    export_parser.add_argument(
        "--format",
        choices=["yaml", "json"],
        default="yaml",
        help="Output format"
    )
    export_parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Exclude metadata from export"
    )
    
    return parser


def cmd_template(args) -> None:
    """Handle template command."""
    try:
        create_config_template(
            output_path=args.output,
            template_type=args.type,
            format=args.format,
            include_comments=not args.no_comments
        )
        print(f"✅ Template created successfully: {args.output}")
    except Exception as e:
        print(f"❌ Error creating template: {e}")
        sys.exit(1)


def cmd_validate(args) -> None:
    """Handle validate command."""
    try:
        # Choose schema
        schema = None
        if args.schema == "refunc":
            schema = RefuncConfig
        
        is_valid = validate_config_file(args.config_file, schema)
        
        if is_valid:
            print(f"✅ Configuration file is valid: {args.config_file}")
        else:
            print(f"❌ Configuration file is invalid: {args.config_file}")
            sys.exit(1)
    
    except Exception as e:
        print(f"❌ Error validating configuration: {e}")
        sys.exit(1)


def cmd_merge(args) -> None:
    """Handle merge command."""
    try:
        merge_config_files(
            input_files=args.input_files,
            output_file=args.output,
            format=args.format
        )
        print(f"✅ Configuration files merged successfully: {args.output}")
    except Exception as e:
        print(f"❌ Error merging configurations: {e}")
        sys.exit(1)


def cmd_show(args) -> None:
    """Handle show command."""
    try:
        # Load configuration
        if args.config_files:
            config = ConfigManager()
            for config_file in args.config_files:
                config.add_file_source(config_file)
        else:
            config = auto_configure()
        
        if args.key:
            # Show specific key
            value = config.get(args.key)
            if value is None:
                print(f"Key '{args.key}' not found")
                sys.exit(1)
            else:
                print(f"{args.key}: {value}")
        else:
            # Show full configuration or summary
            if args.format == "summary":
                summary = get_config_summary(config)
                print("Configuration Summary:")
                print(f"  Sources: {len(summary['sources'])}")
                print(f"  Total Settings: {summary['total_settings']}")
                print(f"  Schema: {summary['schema'] or 'None'}")
                print(f"  Validation: {'Enabled' if summary['validation_enabled'] else 'Disabled'}")
                print(f"  Auto Reload: {'Enabled' if summary['auto_reload'] else 'Disabled'}")
                print("\nSources:")
                for source in summary['sources']:
                    print(f"    {source['name']} (priority: {source['priority']}, format: {source['format']})")
            elif args.format == "json":
                import json
                print(json.dumps(config.to_dict(), indent=2))
            elif args.format == "yaml":
                try:
                    import yaml
                    print(yaml.dump(config.to_dict(), default_flow_style=False, indent=2))
                except ImportError:
                    print("PyYAML not available, falling back to JSON format")
                    import json
                    print(json.dumps(config.to_dict(), indent=2))
    
    except Exception as e:
        print(f"❌ Error showing configuration: {e}")
        sys.exit(1)


def cmd_export(args) -> None:
    """Handle export command."""
    try:
        # Load configuration
        if args.config_files:
            config = ConfigManager()
            for config_file in args.config_files:
                config.add_file_source(config_file)
        else:
            config = auto_configure()
        
        export_config(
            output_path=args.output,
            config=config,
            format=args.format,
            include_metadata=not args.no_metadata
        )
        print(f"✅ Configuration exported successfully: {args.output}")
    
    except Exception as e:
        print(f"❌ Error exporting configuration: {e}")
        sys.exit(1)


def main(argv: Optional[List[str]] = None) -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    if not args.command:
        parser.print_help()
        return
    
    # Route to command handlers
    if args.command == "template":
        cmd_template(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "merge":
        cmd_merge(args)
    elif args.command == "show":
        cmd_show(args)
    elif args.command == "export":
        cmd_export(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()