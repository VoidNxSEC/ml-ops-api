#!/usr/bin/env bash
# =============================================================================
# tensorforge/scripts/setup-shell.sh
# Beautiful, productive shell for GPU inference nodes
#
# Installs:
#   - zsh + zsh-autosuggestions + zsh-syntax-highlighting + zsh-completions
#   - starship prompt (fast, cross-shell, GPU/CUDA aware)
#   - fzf (fuzzy finder — history, files, processes)
#   - bat (better cat), eza (better ls), ripgrep, fd
#   - Custom .zshrc with tensorforge + GPU aliases
#
# Usage:
#   ./scripts/setup-shell.sh
#   ./scripts/setup-shell.sh --dry-run
#   ./scripts/setup-shell.sh --no-starship   # skip starship, use simple prompt
# =============================================================================
set -euo pipefail

DRY_RUN=false
WITH_STARSHIP=true

RED='\033[1;31m'; GRN='\033[1;32m'; YLW='\033[1;33m'; BLU='\033[1;34m'; RST='\033[0m'
BOLD='\033[1m'
log()  { echo -e "${BLU}[setup-shell]${RST} $*"; }
ok()   { echo -e "${GRN}[✓]${RST} $*"; }
warn() { echo -e "${YLW}[!]${RST} $*"; }
die()  { echo -e "${RED}[✗]${RST} $*" >&2; exit 1; }
hr()   { echo -e "${BLU}────────────────────────────────────────────${RST}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)     DRY_RUN=true;       shift ;;
    --no-starship) WITH_STARSHIP=false; shift ;;
    -h|--help)
      echo "Usage: $0 [--dry-run] [--no-starship]"
      exit 0 ;;
    *) die "Unknown flag: $1" ;;
  esac
done

SUDO=""
[[ "$(id -u)" != "0" ]] && command -v sudo &>/dev/null && SUDO="sudo"

run() {
  if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "  ${YLW}dry-run${RST}: $*"
  else
    eval "$@"
  fi
}

hr
echo -e "  ${BOLD}tensorforge — shell setup${RST}"
echo    "  zsh + starship + fzf + bat + eza + gpu aliases"
hr
echo ""

# ── 1. zsh + plugins via apt ──────────────────────────────────────────────────
log "[1/5] Installing zsh + terminal tools..."
run "$SUDO apt-get update -qq"
run "$SUDO apt-get install -y --no-install-recommends \
  zsh \
  zsh-autosuggestions \
  zsh-syntax-highlighting \
  fzf \
  bat \
  ripgrep \
  fd-find \
  tmux \
  curl \
  wget \
  unzip"

# eza não está nos repos do Ubuntu 22.04 — instala via cargo ou binário
if ! command -v eza &>/dev/null; then
  log "Installing eza (modern ls)..."
  run "curl -fsSL https://github.com/eza-community/eza/releases/latest/download/eza_x86_64-unknown-linux-gnu.tar.gz \
    | $SUDO tar -xz -C /usr/local/bin eza 2>/dev/null" \
    || warn "eza install failed — falling back to ls"
fi

ok "zsh + tools installed"

# ── 2. Starship prompt ────────────────────────────────────────────────────────
if [[ "$WITH_STARSHIP" == "true" ]]; then
  log "[2/5] Installing starship prompt..."
  if [[ "$DRY_RUN" == "false" ]]; then
    curl -fsSL https://starship.rs/install.sh | sh -s -- --yes 2>/dev/null \
      && ok "starship installed" \
      || warn "starship install failed — using fallback prompt"
  else
    echo -e "  ${YLW}dry-run${RST}: curl starship install script | sh"
  fi
fi

# ── 3. zsh plugins ────────────────────────────────────────────────────────────
log "[3/5] Setting up zsh plugins..."
ZSH_CUSTOM="${HOME}/.zsh"
run "mkdir -p $ZSH_CUSTOM"

# zsh-autosuggestions (system package symlink or git)
if [[ ! -f "$ZSH_CUSTOM/zsh-autosuggestions/zsh-autosuggestions.zsh" ]]; then
  if [[ -f "/usr/share/zsh-autosuggestions/zsh-autosuggestions.zsh" ]]; then
    run "ln -sf /usr/share/zsh-autosuggestions $ZSH_CUSTOM/zsh-autosuggestions"
  else
    run "git clone --depth=1 https://github.com/zsh-users/zsh-autosuggestions $ZSH_CUSTOM/zsh-autosuggestions 2>/dev/null"
  fi
fi

# zsh-syntax-highlighting
if [[ ! -f "$ZSH_CUSTOM/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh" ]]; then
  if [[ -f "/usr/share/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh" ]]; then
    run "ln -sf /usr/share/zsh-syntax-highlighting $ZSH_CUSTOM/zsh-syntax-highlighting"
  else
    run "git clone --depth=1 https://github.com/zsh-users/zsh-syntax-highlighting $ZSH_CUSTOM/zsh-syntax-highlighting 2>/dev/null"
  fi
fi

# fzf tab completion
if [[ ! -f "$ZSH_CUSTOM/fzf-tab/fzf-tab.zsh" ]]; then
  run "git clone --depth=1 https://github.com/Aloxaf/fzf-tab $ZSH_CUSTOM/fzf-tab 2>/dev/null" \
    || true
fi

ok "zsh plugins ready"

# ── 4. Write .zshrc ────────────────────────────────────────────────────────────
log "[4/5] Writing ~/.zshrc..."

ZSHRC="$HOME/.zshrc"
[[ -f "$ZSHRC" ]] && cp "$ZSHRC" "${ZSHRC}.bak.$(date +%s)" && warn "Backed up existing .zshrc"

if [[ "$DRY_RUN" == "false" ]]; then
cat > "$ZSHRC" <<'ZSHRC_EOF'
# =============================================================================
# tensorforge — .zshrc
# =============================================================================

# ── History ───────────────────────────────────────────────────────────────────
HISTFILE=~/.zsh_history
HISTSIZE=10000
SAVEHIST=10000
setopt HIST_IGNORE_DUPS
setopt HIST_IGNORE_SPACE
setopt SHARE_HISTORY
setopt APPEND_HISTORY

# ── Options ───────────────────────────────────────────────────────────────────
setopt AUTO_CD
setopt CORRECT
setopt COMPLETE_ALIASES
setopt EXTENDED_GLOB
setopt NO_BEEP

# ── Completion ────────────────────────────────────────────────────────────────
autoload -Uz compinit && compinit -i
zstyle ':completion:*' menu select
zstyle ':completion:*' matcher-list 'm:{a-z}={A-Z}'
zstyle ':completion:*' list-colors "${(s.:.)LS_COLORS}"

# ── Plugins ───────────────────────────────────────────────────────────────────
ZSH_CUSTOM="${HOME}/.zsh"

[[ -f "$ZSH_CUSTOM/zsh-autosuggestions/zsh-autosuggestions.zsh" ]] \
  && source "$ZSH_CUSTOM/zsh-autosuggestions/zsh-autosuggestions.zsh"

[[ -f "$ZSH_CUSTOM/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh" ]] \
  && source "$ZSH_CUSTOM/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh"

[[ -f "$ZSH_CUSTOM/fzf-tab/fzf-tab.zsh" ]] \
  && source "$ZSH_CUSTOM/fzf-tab/fzf-tab.zsh"

# fzf keybindings
[[ -f /usr/share/doc/fzf/examples/key-bindings.zsh ]] \
  && source /usr/share/doc/fzf/examples/key-bindings.zsh
[[ -f ~/.fzf.zsh ]] && source ~/.fzf.zsh

# ── PATH ──────────────────────────────────────────────────────────────────────
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/local/cuda/bin:$PATH"

# ── tensorforge env ───────────────────────────────────────────────────────────
_TF_PREFIX="$([[ "$(id -u)" == "0" ]] && echo "/opt/tensorforge/llamacpp" || echo "$HOME/.tensorforge/llamacpp")"
[[ -f "$_TF_PREFIX/env.sh" ]] && source "$_TF_PREFIX/env.sh"

# ── GPU aliases ───────────────────────────────────────────────────────────────
alias gpu='nvidia-smi'
alias gpu-watch='watch -n 1 nvidia-smi'
alias gpu-mem='nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv'
alias gpu-top='nvidia-smi dmon -s um'

# ── tensorforge aliases ───────────────────────────────────────────────────────
alias tf='./tensorforge/scripts/entrypoint.sh'
alias tf-start='./tensorforge/scripts/server.sh start'
alias tf-stop='./tensorforge/scripts/server.sh stop'
alias tf-status='./tensorforge/scripts/server.sh status'
alias tf-health='./tensorforge/scripts/health.sh'
alias tf-logs='./tensorforge/scripts/server.sh logs -f'
alias tf-models='./tensorforge/scripts/model-pull.sh --list'
alias tf-infer='./tensorforge/scripts/infer.sh'

# ── Better defaults ───────────────────────────────────────────────────────────
if command -v eza &>/dev/null; then
  alias ls='eza --icons --group-directories-first'
  alias ll='eza -la --icons --group-directories-first --git'
  alias lt='eza --tree --icons --level=2'
else
  alias ls='ls --color=auto'
  alias ll='ls -lahF --color=auto'
fi

if command -v bat &>/dev/null; then
  alias cat='bat --style=plain'
  alias catp='bat'  # with pager + line numbers
elif command -v batcat &>/dev/null; then
  alias cat='batcat --style=plain'
  alias catp='batcat'
fi

command -v rg &>/dev/null && alias grep='rg'
command -v fd  &>/dev/null && alias find='fd'
command -v fdfind &>/dev/null && alias fd='fdfind'

alias cp='cp -iv'
alias mv='mv -iv'
alias mkdir='mkdir -pv'
alias df='df -h'
alias du='du -sh'
alias free='free -h'
alias ps='ps auxf'
alias ports='ss -tulnp'
alias path='echo $PATH | tr ":" "\n"'

# ── Git shortcuts ─────────────────────────────────────────────────────────────
alias gs='git status -sb'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git log --oneline --graph --decorate -20'
alias gd='git diff'

# ── System ────────────────────────────────────────────────────────────────────
alias reload='source ~/.zshrc'
alias zshrc='${EDITOR:-nano} ~/.zshrc'
alias ..='cd ..'
alias ...='cd ../..'
alias -- -='cd -'

# ── Key bindings ─────────────────────────────────────────────────────────────
bindkey '^[[A' history-substring-search-up   2>/dev/null || true
bindkey '^[[B' history-substring-search-down 2>/dev/null || true
bindkey '^R' fzf-history-widget 2>/dev/null || bindkey '^R' history-incremental-search-backward
bindkey '^[[H' beginning-of-line
bindkey '^[[F' end-of-line
bindkey '^[[3~' delete-char

# ── Starship prompt ───────────────────────────────────────────────────────────
command -v starship &>/dev/null && eval "$(starship init zsh)" || {
  # Fallback: clean minimal prompt with git branch
  autoload -Uz vcs_info
  precmd() { vcs_info }
  zstyle ':vcs_info:git:*' formats ' %F{cyan}(%b)%f'
  setopt PROMPT_SUBST
  PROMPT='%F{blue}%n%f@%F{green}%m%f %F{yellow}%~%f${vcs_info_msg_0_} %F{magenta}❯%f '
}

ZSHRC_EOF
fi

ok ".zshrc written"

# ── 5. Starship config ────────────────────────────────────────────────────────
if [[ "$WITH_STARSHIP" == "true" ]]; then
  log "[5/5] Writing starship config..."
  run "mkdir -p ${HOME}/.config"

  if [[ "$DRY_RUN" == "false" ]]; then
  cat > "${HOME}/.config/starship.toml" <<'STARSHIP_EOF'
# tensorforge — starship prompt config
format = """
[╭─](bold blue)$os$username$hostname$directory$git_branch$git_status$cuda$python$rust$fill$cmd_duration$time
[╰─](bold blue)$character"""

[os]
disabled = false
style = "bold white"

[os.symbols]
Linux = " "
Ubuntu = " "

[username]
style_user = "bold cyan"
style_root = "bold red"
format = "[$user]($style)"
show_always = true

[hostname]
ssh_only = false
format = "@[$hostname](bold green) "
trim_at = "."

[directory]
style = "bold yellow"
truncation_length = 4
truncate_to_repo = true
format = "in [$path]($style)[$read_only]($read_only_style) "

[git_branch]
symbol = " "
style = "bold purple"
format = "on [$symbol$branch]($style) "

[git_status]
format = '([\[$all_status$ahead_behind\]]($style) )'
style = "bold red"
conflicted = "⚡"
ahead = "⬆${count}"
behind = "⬇${count}"
modified = "~${count}"
untracked = "?${count}"
staged = "+${count}"

[python]
symbol = " "
format = "[$symbol$version]($style) "
style = "bold green"
detect_files = ["*.py", "requirements.txt", "pyproject.toml"]

[rust]
symbol = " "
format = "[$symbol$version]($style) "
style = "bold orange"
detect_files = ["Cargo.toml"]

[fill]
symbol = " "

[cmd_duration]
min_time = 2_000
format = "took [$duration](bold yellow) "

[time]
disabled = false
format = "[$time](bold white)"
time_format = "%H:%M"

[character]
success_symbol = "[❯](bold green)"
error_symbol = "[❯](bold red)"
STARSHIP_EOF
  fi

  ok "starship config written"
fi

# ── Set zsh as default shell ───────────────────────────────────────────────────
ZSH_BIN=$(command -v zsh 2>/dev/null || echo "/usr/bin/zsh")
if [[ "$SHELL" != "$ZSH_BIN" ]]; then
  log "Setting zsh as default shell..."
  if [[ "$DRY_RUN" == "false" ]]; then
    chsh -s "$ZSH_BIN" "$(whoami)" 2>/dev/null \
      && ok "Default shell → $ZSH_BIN" \
      || warn "Could not change default shell — run manually: chsh -s $ZSH_BIN"
  else
    echo -e "  ${YLW}dry-run${RST}: chsh -s $ZSH_BIN"
  fi
else
  ok "Already using zsh"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
hr
echo -e "  ${GRN}${BOLD}Shell setup complete!${RST}"
hr
echo ""
echo "  Start zsh now:"
echo "    exec zsh"
echo ""
echo "  GPU aliases ready:"
echo "    gpu           → nvidia-smi"
echo "    gpu-watch     → watch -n1 nvidia-smi"
echo "    tf-status     → tensorforge server status"
echo "    tf-health     → full health check"
echo "    tf-logs       → follow server logs"
echo "    ll            → eza -la (or ls -lahF)"
echo ""
