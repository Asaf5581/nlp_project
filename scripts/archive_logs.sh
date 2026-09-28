#!/bin/bash
mkdir -p crashed_archive
for f in *.err *.out; do
  if [[ -f $f ]]; then
    job_id=$(echo $f | awk -F"_" '{print $NF}' | awk -F"." '{print $1}')
    if [[ $job_id -lt 851087 ]]; then
      mv $f crashed_archive/
    fi
  fi
done
