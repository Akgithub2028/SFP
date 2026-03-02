FROM texlive/texlive:latest-medium

RUN tlmgr repository set https://ftp.math.utah.edu/pub/tex/historic/systems/texlive/2025/tlnet-final \
    && tlmgr install cleveref

WORKDIR /paper
ENTRYPOINT ["latexmk", "-pdf", "main.tex"]
