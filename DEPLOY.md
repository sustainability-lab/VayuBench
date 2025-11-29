# Deploying VayuBench GitHub Pages

Simple guide to deploy the Quarto site.

## Setup

1. **Enable GitHub Pages**:
   - Go to Settings → Pages
   - Source: **GitHub Actions**
   - Save

2. **Push changes**:
   ```bash
   git add .
   git commit -m "Add Quarto site"
   git push origin main
   ```

3. **Monitor deployment**:
   - Go to Actions tab
   - Watch the workflow
   - Site will be live at: `https://USERNAME.github.io/VayuBench/`

## Local Development

```bash
# Install Quarto
# Download from https://quarto.org/

# Preview locally
quarto preview

# Build (optional - CI does this)
quarto render
```

## Making Changes

1. Edit `.qmd` files
2. Preview with `quarto preview`
3. Commit and push
4. GitHub Actions deploys automatically

## File Structure

```
├── index.qmd              # Homepage
├── datasets.qmd           # Datasets
├── categories.qmd         # Categories
├── getting-started.qmd    # Getting started
├── results.qmd            # Results
├── paper.qmd              # Paper
├── _quarto.yml            # Config
└── .github/workflows/     # Auto-deploy
```

## Troubleshooting

- **Site not updating?** Check Actions tab, wait 1-2 minutes
- **Workflow failing?** Read error in Actions → failed job
- **Local preview?** Run `quarto preview`
