#!/bin/bash
# A sample Bash script, by Ryan
echo Hello World!
python3 ./hello_world.py

for i in {1..3}
do
  echo "Welcome $i times"
  python3 ./hello_world.py
done
