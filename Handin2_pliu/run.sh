#!/bin/bash

echo "Run handin Exercise 2"

echo "Creating the plotting directory if it does not exist"
if [ ! -d "plots" ]; then
  echo "Directory does not exist create it!"
  mkdir plots
fi

#question1

# Script that returns a plot and a movie
echo "Run the first script question1.py"
python3 question1.py



#question2

#Scripts for question2 Redshift distribution of galaxies
echo "Run the second script question2.py"
python3 question2.py




echo "Generating the pdf"

pdflatex all.tex
