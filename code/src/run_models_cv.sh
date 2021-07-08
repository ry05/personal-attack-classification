echo "Running models created"
echo "----------------------\n"

echo "\nGroup 1: Models with raw data"
python train2.py --pipeline_number 1 --data raw
python train2.py --pipeline_number 2 --data raw
python train2.py --pipeline_number 3 --data raw

echo "\n----------------------\n"

echo "\n\nGroup 2: Models with data with numeric features included"
python train2.py --pipeline_number 4 --data with_numeric
python train2.py --pipeline_number 4 --data with_numeric_translated_text

echo "\nComplete."