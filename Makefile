.PHONEY: serve




serve:
	quarto preview ./index.qmd

publish:
	uv run quarto publish gh-pages
