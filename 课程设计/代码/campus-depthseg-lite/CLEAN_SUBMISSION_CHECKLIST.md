# Clean Submission Checklist

Use a clean archive for the final course handoff. Do not provide a Git repository with local history or generated artifacts.

## Include

- `README.md`
- `requirements.txt`
- `EXPERIMENT_RESULTS.md`
- `src/`
- `scripts/`
- `tests/`
- Final report document, if placed in the clean export folder
- Small static demo assets only if required by the course submission

## Exclude

- `.git/`
- `data/`
- `outputs/`
- `checkpoints/`
- `lightning_logs/`
- `wandb/`
- `__pycache__/`
- `.pytest_cache/`
- `*.ckpt`
- `*.pt`
- `*.pth`
- `*.mat`
- `*.npy`
- `CLEAN_ROOM_AUDIT.md`
- `CLEAN_ROOM_FIX_PLAN.md`
- Local absolute paths or machine-specific notes

## Export Command Template

Replace `<project_root>` and `<clean_root>` with local paths before running:

```bat
robocopy "<project_root>" "<clean_root>" ^
/E ^
/XD .git data outputs checkpoints lightning_logs wandb __pycache__ .pytest_cache ^
/XF *.ckpt *.pt *.pth *.mat *.npy *.zip CLEAN_ROOM_AUDIT.md CLEAN_ROOM_FIX_PLAN.md
```

Then compress `<clean_root>` as the final course-design archive.

## Final Handoff Recommendation

- Include the final report document.
- Include the clean archive generated from the template above.
- Do not include generated data, logs, checkpoints, local audit artifacts, or a repository with full local history.
