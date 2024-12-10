obtain_data:
	echo Obtaining raw data...
	mkdir -p src
	curl -L -o src2/data.zip https://www.kaggle.com/api/v1/datasets/download/cdminix/us-drought-meteorological-data
	unzip src2/data.zip -d src
	rm src2/data.zip
	echo Raw data obtained and saved in src directory!

reset_data:
	echo Removing data/processed directory...
	rm -rf src
	mkdir -p src
	echo src directory reset!
