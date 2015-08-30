#!/bin/bash

ffmpeg -f image2 -i "./out/image/frame_%05d.png" -r 24 $1
