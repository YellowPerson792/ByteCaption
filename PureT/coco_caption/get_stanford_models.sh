#!/usr/bin/env sh
# This script downloads the Stanford CoreNLP models.

CORENLP=stanford-corenlp-full-2015-12-09
SPICELIB=pycocoevalcap/spice/lib
JAR=stanford-corenlp-3.6.0
MODELS_JAR=${JAR}-models.jar

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

if [ -f "$SPICELIB/$JAR.jar" ] && [ -f "$SPICELIB/$MODELS_JAR" ]; then
  echo "Found Stanford CoreNLP and models jar."
else
  echo "Downloading..."
  wget http://nlp.stanford.edu/software/$CORENLP.zip
  echo "Unzipping..."
  unzip $CORENLP.zip -d $SPICELIB/
  mv $SPICELIB/$CORENLP/$JAR.jar $SPICELIB/
  mv "$SPICELIB/$CORENLP/$JAR-models.jar" "$SPICELIB/" || true
  mv "$SPICELIB/$CORENLP/stanford-corenlp-3.6.0-models.jar" "$SPICELIB/" 2>/dev/null || true
  rm -f $CORENLP.zip
  rm -rf $SPICELIB/$CORENLP/
  echo "Done."
fi
