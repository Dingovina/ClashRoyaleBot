# Data Directory

- `raw/` keeps original captures when needed.
- `processed/` keeps compact structured datasets for training.
- Raw dataset layout:
  - `raw/battlefield_test/good/` and `raw/battlefield_test/bad/`
  - `raw/elixir_test/`
- Processed dataset layout (used by training):
  - `processed/battlefield_test/good/` and `processed/battlefield_test/bad/`
  - `processed/elixir_test/`

Default policy: prefer compact processed artifacts over full raw video.
