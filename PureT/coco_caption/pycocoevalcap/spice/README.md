The Stanford CoreNLP models archive (`stanford-corenlp-3.6.0-models.jar`) is not
committed to this repository to keep the repo under GitHub's file size limits.
Download it from https://stanfordnlp.github.io/CoreNLP/ and place it under `lib/` or
use the included helper script to download it for you:

```bash
cd PureT/coco_caption
bash get_stanford_models.sh
```

Note: the original `get_stanford_models.sh` only checked for the presence of
the CoreNLP jar (without the models). If you still encounter the 'Unable to
open "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz"' error, make sure
that `stanford-corenlp-3.6.0-models.jar` exists under `pycocoevalcap/spice/lib/`.

Optional: for faster repeated evaluations, a persistent SPICE server can be
used to keep the Stanford CoreNLP pipeline warm. If compiled classes are
available under `server_classes/`, the Python wrapper will auto-use it. To
force server compilation/use, set:

```bash
export SPICE_SERVER=1
```
