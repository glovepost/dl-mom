FROM pandoc/latex:3

# Install SVG renderer and xelatex on the Alpine-based image.
RUN apk add --no-cache \
    librsvg \
    texlive-xetex \
    fontconfig \
    ttf-dejavu \
    ttf-liberation \
    font-noto \
    font-noto-cjk && \
    fc-cache -f
