BIB = --bibliography ml-2019.bib
STYLE = --csl https://raw.githubusercontent.com/citation-style-language/styles/master/nature.csl

report: report.md
	pandoc ${BIB} ${STYLE} -o $@.pdf $^

.PHONY: clean
clean:
	-rm report.pdf
