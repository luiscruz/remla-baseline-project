for f in *.py; do docformatter --in-place $f; done
black src