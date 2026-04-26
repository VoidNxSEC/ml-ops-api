#!/usr/bin/env bash
cat << 'INNER_EOF'
{
  "systemMessage": "👋 **Welcome to the ML Offload API & TensorForge Workspace!**\n\n**🔧 Quick Project Commands:**\n- `/just dev`: Enter the Nix development shell (Rust + CUDA)\n- `/just dev-python`: Enter the Python Nix shell\n- `/just setup-linux`: Bare Linux setup\n- `/just quick-nix`: Quick Nix dev shell\n\n**🤖 Gemini CLI Commands:**\n- `/help`: View all available commands\n- `/memory show`: See current session context\n- `/clear`: Clear the session"
}
INNER_EOF
