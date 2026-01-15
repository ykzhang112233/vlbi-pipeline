#!/bin/bash
# Example script showing how to run the pipeline with different configurations

echo "============================================"
echo "VLBI Pipeline Configuration Examples"
echo "============================================"
echo

# Example 1: Using --config parameter (Recommended)
echo "Example 1: Using --config parameter"
echo "--------------------------------------------"
echo "Command:"
echo "  ParselTongue main.py --config configs/ba158l1_input.py"
echo
echo "This is the recommended way to specify configuration."
echo

# Example 2: Using environment variable
echo "Example 2: Using environment variable"
echo "--------------------------------------------"
echo "Commands:"
echo "  export VLBI_CONFIG=configs/bz111cl_input.py"
echo "  ParselTongue main.py"
echo
echo "Useful for batch processing or when you want to"
echo "set the config once and run multiple commands."
echo

# Example 3: Using relative path
echo "Example 3: Using relative path"
echo "--------------------------------------------"
echo "Command (from project root):"
echo "  cd vlbi-pipeline"
echo "  ParselTongue main.py --config ../configs/ba158l1_input.py"
echo

# Example 4: Using absolute path
echo "Example 4: Using absolute path"
echo "--------------------------------------------"
echo "Command:"
echo "  ParselTongue main.py --config /full/path/to/configs/ba158l1_input.py"
echo

# Example 5: Batch processing
echo "Example 5: Batch processing multiple observations"
echo "--------------------------------------------"
echo "Script:"
cat << 'EOF'
  #!/bin/bash
  for config in configs/*_input.py; do
      if [[ "$config" != "configs/template_input.py" ]]; then
          echo "Processing $config..."
          ParselTongue main.py --config "$config"
      fi
  done
EOF
echo

# Quick test
echo "============================================"
echo "Quick Configuration Test"
echo "============================================"
echo
echo "To test if configuration loading works:"
echo "  python test_config_loading.py"
echo

echo "============================================"
echo "For more information, see:"
echo "  - configs/README.md"
echo "  - configs/USAGE.md"
echo "  - configs/template_input.py"
echo "============================================"
