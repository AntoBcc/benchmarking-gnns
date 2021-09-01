# Command to download dataset:
#   bash script_download_TSP.sh

DIR=TSP/
cd $DIR

FILE=tsp30-50.zip
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
    curl https://www.dropbox.com/s/3wtbbunpoyd8eor/tsp30-50.zip?dl=0 -o tsp30-50.zip -J -L -k
fi

FILE=tsp30-50.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://www.dropbox.com/s/42kn8qwwz0vc0bs/tsp30-50.pkl?dl=0 -o $FILE -J -L -k
fi

FILE=tsp50-500.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://www.dropbox.com/s/qga6q0gxx3wb8k0/TSP.pkl?dl=1 -o $FILE -J -L -k
fi



