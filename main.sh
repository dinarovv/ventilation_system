#!/bin/bash

clear
BAT_DIR="$(dirname "$(realpath "$0")")/"

pip install -r "${BAT_DIR}requirements.txt" > /dev/null
clear
python3 "${BAT_DIR}src/main.py"

read -p "Нажмите Enter для продолжения..."
clear
exit 0