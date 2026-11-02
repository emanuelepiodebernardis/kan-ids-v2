.PHONY: all figures clean
all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
figures:
	python build_assets.py
clean:
	latexmk -c main.tex
	latexmk -c supplement.tex
