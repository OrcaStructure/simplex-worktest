# Research Report (LaTeX)

This folder contains a professional LaTeX report template.

## Structure

- `main.tex`: main report file
- `references.bib`: bibliography file
- `figures/`: place images/plots here
- `sections/`: optional split chapter files

## Build

Use one of the following commands from this directory:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

or (recommended if installed):

```bash
latexmk -pdf main.tex
```

## Quick start edits

1. Update title, author, and abstract in `main.tex`.
2. Replace section placeholder text with your content.
3. Add citation entries to `references.bib` and cite with `\citep{...}`.
4. Put figures in `figures/` and include via `\includegraphics`.
