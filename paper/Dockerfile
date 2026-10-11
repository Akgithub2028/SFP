FROM texlive/texlive:latest-medium

RUN tlmgr --repository https://mirror.ox.ac.uk/sites/ctan.org/systems/texlive/tlnet install cleveref

WORKDIR /paper
ENTRYPOINT ["latexmk", "-pdf", "main.tex"]
