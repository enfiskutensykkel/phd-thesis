# The main tex file must be first in list
# The generated PDF will have the same name
TEXFILES := main.tex glossary.tex $(wildcard sections/*.tex) $(wildcard stubs/*.tex)

DELIVERY := Markussen2022

FIGURES := $(wildcard figures/*.pdf)

# Name of the PDF to create
MAIN = $(firstword $(TEXFILES:%.tex=%))

# How to invoke pdflatex
OPTIONS := -halt-on-error -synctex=1

ifndef VERBOSE
OPTIONS += -interaction=batchmode
endif

.PHONY: default clean distclean 


default: $(MAIN).pdf 
	

$(MAIN).pdf: $(MAIN).bbl $(MAIN).gls $(TEXFILES) $(FIGURES)
	$(info **** MAIN DOCUMENT ****)
	pdflatex $(OPTIONS) $(MAIN)


$(MAIN).gls: glossary.tex
	$(info **** GLOSSARY ****)
	pdflatex $(OPTIONS) $(MAIN)
	makeglossaries -q $(MAIN)
	pdflatex $(OPTIONS) $(MAIN)
	makeglossaries -q $(MAIN)


$(MAIN).bbl: bibliography.bib papers.bib
	$(info **** BIBLIOGRAPHY ****)
	pdflatex $(OPTIONS) $(MAIN)
	biber $(MAIN)
	pdflatex $(OPTIONS) $(MAIN)


$(MAIN).info: $(MAIN).pdf
	pdftk $< dump_data > $(MAIN).info


$(DELIVERY).pdf: $(MAIN).pdf $(MAIN).info
	pdftk $< cat 1-r2 output temp.pdf
	pdftk temp.pdf update_info $(MAIN).info output $@


#$(DELIVERY)-a4.pdf: $(DELIVERY).pdf $(MAIN).info
#	pdfjam -o $@ --paper a4paper -- $<
#	#pdfjam -o temp-a4.pdf --paper a4paper -- $<
#	#pdftk temp-a4.pdf update_info $(MAIN).info output $@

clean:
	-$(RM) $(TEXFILES:%.tex=%.aux)
	-$(RM) $(MAIN).idx $(MAIN).ind $(MAIN).glo $(MAIN).brf $(MAIN).ilg $(MAIN).ist $(MAIN).nlo $(MAIN).nls $(MAIN).acn $(MAIN).gls $(MAIN).glg
	-$(RM) $(MAIN).log $(MAIN).bbl $(MAIN).blg $(MAIN).dvi $(MAIN).bak $(MAIN).toc $(MAIN).ps $(MAIN).out $(MAIN).lof $(MAIN).lot
	-$(RM) $(MAIN).alg $(MAIN).acr $(MAIN).loa $(MAIN).lol $(MAIN).cut $(MAIN).bcf $(MAIN).nlg $(MAIN).ptc $(MAIN).nolist*
	-$(RM) $(MAIN).run.xml $(MAIN).synctex.gz $(MAIN).pdfsync
	-$(RM) $(MAIN)-blx.bib
	-$(RM) $(MAIN).info
	-$(RM) temp.pdf temp-a4.pdf


distclean: clean
	-$(RM) $(MAIN).pdf
	-$(RM) *~

