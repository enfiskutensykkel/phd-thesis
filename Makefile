TEXFILE := main

.PHONY: all $(TEXFILE).pdf clean distclean

all: $(TEXFILE).pdf


$(TEXFILE).pdf:
	pdflatex -synctex=1 $(TEXFILE)
	makeglossaries $(TEXFILE)
	pdflatex -synctex=1 $(TEXFILE)
	pdflatex -synctex=1 $(TEXFILE)


distclean: clean
	-$(RM) $(TEXFILE).pdf


clean:
	-$(RM) *.idx *.ind *.glo *.brf *.ilg *.ist *.nlo *.nls *.acn *.gls *.glg *.glg
	-$(RM) *.log *.aux sections/*.aux *.bbl *.blg *.dvi *.bak *.toc *.ps *.synctex.gz *.pdfsync *.out *.lof *.lot
	-$(RM) *.alg *.acr *.loa *.lol *.cut *.bcf *.run.xml *.nlg *.ptc
	-$(RM) $(TEXFILE)-blx.bib
	-$(RM) *~
	
