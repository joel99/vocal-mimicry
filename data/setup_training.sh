# Script for fetching data and models necessary for training
wget -O raw/VCTK_corpus.zip https://datashare.is.ed.ac.uk/bitstream/handle/10283/2651/VCTK-Corpus.zip?sequence=2&isAllowed=y
cd raw
unzip VCTK_corpus.zip
rm VCTK_corpus.zip
cd ..
