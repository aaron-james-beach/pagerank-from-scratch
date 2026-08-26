Place the edge list here.

    curl -O https://snap.stanford.edu/data/web-Google.txt.gz
    gunzip web-Google.txt.gz
    mv web-Google.txt data/

Any whitespace-separated edge list of integer node ids works. Lines
beginning with `#` are skipped.
