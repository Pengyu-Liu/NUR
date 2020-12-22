#!/bin/bash

echo "Run handin Exercise 1"

echo "Creating the plotting directory if it does not exist"
if [ ! -d "plots" ]; then
  echo "Directory does not exist create it!"
  mkdir plots
fi

#question1
echo "Check if the cooling_rate movie exist"
if [ -e cooling_ratemovie.mp4 ]; then
  echo "Remove mp4 file" -r
  rm cooling_ratemovie.mp4
fi

echo "Download coolingtables for calculation and interpolation in report cooling_rate.pdf"
if [ ! -d 'CoolingTables' ]; then
  wget https://www.strw.leidenuniv.nl/WSS08/coolingtables_highres.tar.gz
  #untar the file
  tar -zxvf coolingtables_highres.tar.gz
fi

# Script that returns a plot and a movie
echo "Run the first script cooling.py"
python3 cooling.py


# code that makes a movie of the movie frames
ffmpeg -framerate 10 -pattern_type glob -i "plots/snap*.png" -s:v 640x480 -c:v libx264 -profile:v high -level 4.0 -crf 10 -tune animation -preset slow -pix_fmt yuv420p -r 25 -threads 0 -f mp4 cooling_ratemovie.mp4

#question2
echo "Download wgs and wss data for question2"
if [ ! -e wgs.dat ]; then
  wget https://home.strw.leidenuniv.nl/~daalen/Handin_files/wgs.dat
  wget https://home.strw.leidenuniv.nl/~daalen/Handin_files/wss.dat
fi

#Scripts for question2 Redshift distribution of galaxies
echo "Run the second script solveLU2.py for question2"
python3 solveLU2.py


#question3
echo "RUn the third script 3integral.py for question3"
python3 3integral.py



echo "Generating the pdf"

pdflatex all.tex
#bibtex all.aux
#pdflatex cooling1.tex
#pdflatex LU2.tex
#pdflatex intergal3.tex
