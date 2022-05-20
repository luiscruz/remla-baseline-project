pip install "dvc[gdrive]"

# Check if pip install worked, if not then try pip3
if [ $? -eq 0 ]; then
   echo "pip install succeeded. DVC with gdrive support installed successfully!";
else
   pip3 install "dvc[gdrive]";
fi