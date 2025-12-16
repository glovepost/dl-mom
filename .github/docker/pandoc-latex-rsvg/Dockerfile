FROM pandoc/latex:3

# Install SVG renderer and xelatex on the Alpine-based image.
RUN apk add --no-cache \
    librsvg \
    texlive-xetex \
    fontconfig \
    ttf-dejavu \
    ttf-liberation \
    font-noto && \
    fc-cache -f
