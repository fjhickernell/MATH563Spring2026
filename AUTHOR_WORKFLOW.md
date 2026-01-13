# MATH Course Quarto Website — Author Workflow

This document is a reminder to **the course author (me)** of how this
Quarto website is structured and how to safely edit, preview, and publish it.

This repo uses **Quarto + GitHub Pages (gh-pages)** with **no rendered output
tracked on `main`**.

---

## 1. High-level structure

- `main` branch  
  - Source only: `.qmd`, YAML, CSS, JS, assets  
  - **No rendered HTML committed**
- `gh-pages` branch  
  - Rendered site (managed automatically by Quarto)
- GitHub Pages  
  - Deploys from **`gh-pages` / root**

---

## 2. Where things live

### Pages
- `index.qmd` — landing page
- `pages/*.qmd` — syllabus, schedule, homework, tests, resources

### Slides
- `slides/*.qmd` — RevealJS slides  
- Slide styling and macros come from **HickernellClassLib** via `metadata-files`

### Shared styling (do not edit locally)
Provided by the `classlib` submodule:
- `classlib/classlib/quarto/website/hickernell-website.yml`
- `classlib/classlib/quarto/website/hickernell-website.css`
- `classlib/classlib/quarto/slides/hickernell-slides.css`

---

## 3. Python / reticulate setup (important)

Quarto renders Python chunks via **reticulate** using the `qmcpy` conda env.

This is enforced in `_quarto.yml`:

```yaml
execute:
  env:
    RETICULATE_PYTHON: /Users/fredhickernell/miniconda3/envs/qmcpy/bin/python
```

If Python-related render errors appear, **check this first**.

---

## 4. Editing workflow

1. Edit `.qmd` files (pages or slides)
2. Edit shared assets only via **HickernellClassLib**, not copied files
3. Save changes

No Git operations are required just to preview.

---

## 5. Preview locally (safe, non-destructive)

```bash
quarto preview
```

- Live reload
- Uses local browser
- Does **not** affect Git or GitHub Pages

Use this for day-to-day editing.

---

## 6. Publish the site (this is the only deploy command)

```bash
quarto publish gh-pages --no-browser
```

What this does:
- Renders the site
- Switches internally to `gh-pages`
- Commits rendered output there
- Pushes to GitHub
- Returns you to `main`

**Do NOT**:
- `git add docs`
- `git commit` rendered HTML
- manually switch branches

---

## 7. After publishing

- GitHub Pages may take 30–120 seconds
- Hard refresh the site: **⌘⇧R**
- If needed, open in a private window to bypass caching

Live site:  
https://fjhickernell.github.io/MATH476Spring2026/

---

## 8. Submodules (important for fresh clones)

This repo depends on submodules:
- `classlib` (HickernellClassLib)
- `qmcsoftware`

After a fresh clone, always run:

```bash
git submodule update --init --recursive
```

If shared CSS/YAML files are reported missing during render, this is the
first thing to check.

---

## 9. Things that are intentionally NOT done

- No `docs/` deployment
- No rendered files tracked on `main`
- No GitHub Actions for Pages
- No `quarto publish` from inside `gh-pages`

This keeps the repo clean and predictable.

---

## 10. Minimal checklist (TL;DR)

- Edit files
- `quarto preview`
- When ready:
  ```bash
  quarto publish gh-pages --no-browser
  ```
- Refresh the website

That’s it.
