#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "There must be 2 parameters"
exit
fi


orig_file="$(pwd)/src/orig.prototxt"
new_file="$(pwd)/src/current.prototxt"

cat $orig_file > $new_file

sed -i s/X_DIM/$1/g $new_file
sed -i s/Y_DIM/$2/g $new_file
