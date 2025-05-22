#!/bin/bash -e
if aplay -l 2>&1 | grep -qi "no soundcards found"; then
    echo "Error: No sound playback devices found." >&2
    exit 1
fi

cd /home/dietpi/festivity
. env/bin/activate
amixer -c 0 set 'Speaker' 100%

while true; do
    ./pre_game.py
    ./festivity.py
done
