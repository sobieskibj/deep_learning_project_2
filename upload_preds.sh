for filename in results/*
do
    kaggle competitions submit -c tensorflow-speech-recognition-challenge -f $filename -m "Message"
done