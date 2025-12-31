#!/usr/bin/env bash
set -euo pipefail

rm -rf docs
quarto render

mkdir -p docs/classlib/classlib/quarto/slides
mkdir -p docs/classlib/classlib/quarto/assets/images

cp -f classlib/classlib/quarto/slides/hickernell-slides.css \
      docs/classlib/classlib/quarto/slides/hickernell-slides.css

cp -f classlib/classlib/quarto/slides/hickernell-latex-macros.js \
      docs/classlib/classlib/quarto/slides/hickernell-latex-macros.js

cp -f classlib/classlib/quarto/assets/images/normal-scatter.png \
      docs/classlib/classlib/quarto/assets/images/normal-scatter.png

test -f docs/classlib/classlib/quarto/slides/hickernell-slides.css
test -f docs/classlib/classlib/quarto/slides/hickernell-latex-macros.js
test -f docs/classlib/classlib/quarto/assets/images/normal-scatter.png
