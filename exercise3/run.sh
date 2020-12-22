#!/bin/bash

echo "Run handin Exercise 3"

echo "Creating the plotting directory if it does not exist"
if [ ! -d "plots" ]; then
  echo "Directory does not exist create it!"
  mkdir plots
fi

#question1
echo "Download satellites for question1"
if [ ! -e satgals_m11.txt ]; then
  wget https://home.strw.leidenuniv.nl/~daalen/files/satgals_m11.txt
  wget https://home.strw.leidenuniv.nl/~daalen/files/satgals_m12.txt
  wget https://home.strw.leidenuniv.nl/~daalen/files/satgals_m13.txt
  wget https://home.strw.leidenuniv.nl/~daalen/files/satgals_m14.txt
  wget https://home.strw.leidenuniv.nl/~daalen/files/satgals_m15.txt
fi
# Script that returns a plot and a movie
echo "Run the first script question1.py"
python3 question1.py


#question2

#Scripts for question2 Redshift distribution of galaxies
echo "Run the second script question2.py"
python3 question2.py


echo "Generating the pdf"

pdflatex all.tex
