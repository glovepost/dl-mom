FROM pandoc/latex:3

# Install SVG renderer and xelatex on the Alpine-based image.
RUN apk add --no-cache \
    librsvg \
    texlive-xetex \
    fontconfig \
    ttf-dejavu \
    ttf-dejavu-sans-mono \
    ttf-dejavu-serif \
    ttf-liberation && \
    fc-cache -f
