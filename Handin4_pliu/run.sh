#!/bin/bash

echo "Run handin Exercise 4"

echo "Creating the plotting directory if it does not exist"
if [ ! -d "plots" ]; then
  echo "Directory does not exist create it!"
  mkdir plots
fi

#question1
echo "Run the first script question1.py"
python3 question1.py


#question2
echo "Download galaxy data for question2"
if [ ! -e galaxy_data.txt ]; then
  wget https://home.strw.leidenuniv.nl/~garcia/NUR/galaxy_data.txt
fi

echo "Run the second script question2.py"
python3 question2.py

echo "Generating the pdf"

pdflatex all.tex
