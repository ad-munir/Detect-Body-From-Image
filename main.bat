cd body
python body_detection.py ../image.jpg
TIMEOUT 2
cd ..
cd age
python age_detection.py ../toDelete.jpg
TIMEOUT 2
cd ..
cd gender
python gender_detection.py ../toDelete.jpg
cd ..
del "toDelete.jpg"
pause