# Deep-Latent Mixture of Models (DL-MoM) - Project Page

This is the academic project page for **Deep-Latent Mixture of Models (DL-MoM): A Training-Free Architecture for System 2 Reasoning via Latent-Space Collaboration**.

## 🚀 Quick Start

1. Clone this repository
2. Open `index.html` in your browser, or
3. Deploy to GitHub Pages

### GitHub Pages Deployment

1. Go to your repository Settings
2. Navigate to Pages
3. Select "Deploy from a branch"
4. Choose `main` branch and `/ (root)` folder
5. Click Save

Your site will be live at `https://yourusername.github.io/dl-mom/`

## 📁 Project Structure

```
dl-mom-website/
├── index.html              # Main project page
├── docs/
│   └── dl-mom-paper.md      # Paper (Markdown)
├── runs/
│   └── ablations/
│       └── plots/           # Experiment figures (PNG)
├── static/
│   ├── css/
│   │   └── style.css       # Custom styles
│   ├── images/
│   │   ├── architecture.svg    # Architecture diagram
│   │   ├── favicon.svg         # Site favicon
│   │   └── social_preview.png  # Social media preview (add your own)
│   ├── pdfs/
│   │   └── dl-mom-paper.pdf    # Paper PDF (add your own)
│   └── videos/                 # Video assets (optional)
├── .nojekyll               # Disable Jekyll processing
└── README.md               # This file
```

## 🧪 Running Ablations on AMD Halo Strix (gfx1151)

The ablation runner is specified in Section 6 of the paper (`docs/dl-mom-paper.md`). On AMD Halo Strix (gfx1151),
use ROCm/HIP and run the suites with the `dlmom` CLI.

### Prerequisites

1. ROCm/HIP installed and visible (sanity check with `rocminfo` and `rocm-smi`).
2. `dlmom` CLI available in your environment (see the paper's runner spec).

### Example: Run a Suite

```bash
dlmom run \
  --suite A1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --bench gsm8k \
  --out runs/ablations \
  --device hip \
  --dtype bf16 \
  --seeds 0 1 2 \
  --max-steps 256 \
  --max-handoffs 2 \
  --max-switches 10 \
  --batch-size auto \
  --resume
```

### Aggregate and Plot

```bash
dlmom aggregate \
  --in runs/ablations \
  --out runs/ablations/aggregate.csv
```

```bash
dlmom plot \
  --in runs/ablations \
  --out runs/ablations/plots \
  --format png
```

### Notes

- Use `--device hip` and `--dtype bf16` for ROCm/HIP.
- Keep `--max-steps`, `--max-handoffs`, and `--max-switches` fixed across suites for comparability.
- The plotting output is expected at `runs/ablations/plots` and is embedded in the project page.

## ✏️ Customization

### Adding Your Paper PDF
Place your paper at `static/pdfs/dl-mom-paper.pdf`

### Adding Social Preview Image
Create a 1200x630px image at `static/images/social_preview.png` for optimal social media sharing.

### Updating Links
Edit `index.html` and replace:
- arXiv link: `https://arxiv.org/abs/XXXX.XXXXX`
- GitHub link: `https://github.com/anonymous/dl-mom`
- Author information and affiliations

### Adding Videos
1. Place video files in `static/videos/`
2. Add video sections in `index.html` following the template structure

## 🎨 Features

- **Responsive Design**: Works on desktop and mobile
- **Modern Styling**: Clean, academic aesthetic with gradient accents
- **Interactive Elements**:
  - "More Works" dropdown for related papers
  - Copy-to-clipboard for BibTeX
  - Smooth hover effects
- **SEO Optimized**: Meta tags for Google Scholar and social media
- **MathJax Support**: Render LaTeX equations
- **Accessibility**: Semantic HTML and proper contrast

## 📚 Based On

This project page is built using the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template) by Eliahu Horwitz.

## 📜 License

This website is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).

## 📖 Citation

```bibtex
@article{dlmom2025,
  title={Deep-Latent Mixture of Models: A Training-Free Architecture 
         for System 2 Reasoning via Latent-Space Collaboration},
  author={Anonymous},
  journal={Under Review},
  year={2025},
  url={https://github.com/anonymous/dl-mom}
}
```
