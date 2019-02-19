wget http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar
tar -xvf places365standard_easyformat.tar
rm places365standard_easyformat.tar
ls -l places365_standard/train | egrep '^d' | awk '{print $9}' > classes.txt
