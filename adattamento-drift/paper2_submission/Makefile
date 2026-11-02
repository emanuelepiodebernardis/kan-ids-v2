.PHONY: all clean
all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error author-version.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error supplement-author.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error editor-letter.tex
clean:
	latexmk -c main.tex supplement.tex author-version.tex supplement-author.tex editor-letter.tex
