# The main tex file must be first in list
# The generated PDF will have the same name
TEXFILES := main.tex glossary.tex $(wildcard sections/*.tex) $(wildcard stubs/*.tex)

# Name of the PDF to create
MAIN = $(firstword $(TEXFILES:%.tex=%))

.PHONY: default clean distclean


default: $(MAIN).pdf


$(MAIN).pdf: $(TEXFILES) bibliography.bib
	pdflatex -synctex=1 $(MAIN)
	pdflatex -synctex=1 $(MAIN)
	biber $(MAIN)
	makeglossaries $(MAIN)
	pdflatex -synctex=1 $(MAIN)
	makeglossaries $(MAIN)
	pdflatex -synctex=1 $(MAIN)


clean:
	-$(RM) $(TEXFILES:%.tex=%.aux)
	-$(RM) $(MAIN).idx $(MAIN).ind $(MAIN).glo $(MAIN).brf $(MAIN).ilg $(MAIN).ist $(MAIN).nlo $(MAIN).nls $(MAIN).acn $(MAIN).gls $(MAIN).glg
	-$(RM) $(MAIN).log $(MAIN).bbl $(MAIN).blg $(MAIN).dvi $(MAIN).bak $(MAIN).toc $(MAIN).ps $(MAIN).out $(MAIN).lof $(MAIN).lot
	-$(RM) $(MAIN).alg $(MAIN).acr $(MAIN).loa $(MAIN).lol $(MAIN).cut $(MAIN).bcf $(MAIN).nlg $(MAIN).ptc $(MAIN).nolist*
	-$(RM) $(MAIN).run.xml $(MAIN).synctex.gz $(MAIN).pdfsync
	-$(RM) $(MAIN)-blx.bib


distclean: clean
	-$(RM) $(MAIN).pdf
	-$(RM) *~

