all: all_pdfs all_pngs clean

all_pdfs:
	pdflatex linreg1.tex
	pdflatex linreg2.tex
	pdflatex logreg1.tex
	pdflatex logreg2.tex
	pdflatex nn.tex

all_pngs:
	convert -density 300 linreg1.pdf -quality 90 pngs/linreg1.png
	convert -density 300 linreg2.pdf -quality 90 pngs/linreg2.png
	convert -density 300 logreg1.pdf -quality 90 pngs/logreg1.png
	convert -density 300 logreg2.pdf -quality 90 pngs/logreg2.png
	convert -density 300 nn.pdf -quality 90 pngs/nn.png
	convert -density 300 linreg3.pdf -quality 90 pngs/linreg3.png

clean:
	rm *.pdf; rm *.log; rm *.aux


