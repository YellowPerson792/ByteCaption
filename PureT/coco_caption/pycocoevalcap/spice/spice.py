from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import subprocess
import threading
import json
import numpy as np
import ast
import tempfile

import time
import shutil

# Assumes spice.jar is in the same directory as spice.py.  Change as needed.
SPICE_JAR = 'spice-1.0.jar'
TEMP_DIR = 'tmp'
CACHE_DIR = 'cache'

class Spice:
    """
    Main Class to compute the SPICE metric 
    """
    def __init__(self):
        cwd = os.path.dirname(os.path.abspath(__file__))
        cache_dir_env = os.environ.get("SPICE_CACHE_DIR", "").strip()
        if cache_dir_env:
          cache_dir = cache_dir_env
        else:
          # Use a stable cache directory to reuse parsed references across runs
          cache_dir = os.path.join(cwd, CACHE_DIR, "shared")
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
          os.makedirs(cache_dir)

    def float_convert(self, obj):
        try:
          return float(obj)
        except:
          return np.nan

    def compute_score(self, gts, res):
        assert(sorted(gts.keys()) == sorted(res.keys()))
        imgIds = sorted(gts.keys())
        
        # Prepare temp input file for the SPICE scorer
        input_data = []
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) >= 1)
            assert(type(ref) is list)
            assert(len(ref) >= 1)

            input_data.append({
              "image_id" : id,
              "tests" : hypo,
              "refs" : ref
            })

        cwd = os.path.dirname(os.path.abspath(__file__))
        temp_dir=os.path.join(cwd, TEMP_DIR)
        if not os.path.exists(temp_dir):
          os.makedirs(temp_dir)
        in_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
        in_file.write(json.dumps(input_data, indent=2).encode('utf-8'))
        in_file.close()

        # Start job
        out_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
        out_file.close()
        # Ensure that the required Stanford CoreNLP models jar is present next to spice-1.0.jar
        # This avoids the 'Unable to open edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz' exception
        exists_model = False
        # The models jar might be placed either in the current directory
        # or under the bundled lib/ folder (as shipped in this repo).
        search_roots = [cwd, os.path.join(cwd, 'lib')]
        for root in search_roots:
          if not os.path.isdir(root):
            continue
          for fname in os.listdir(root):
            if fname.endswith('-models.jar') and fname.startswith('stanford-corenlp'):
              exists_model = True
              break
          if exists_model:
            break
        if not exists_model:
          raise RuntimeError('\nSPICE evaluation requires the Stanford CoreNLP models jar (e.g. stanford-corenlp-3.6.0-models.jar) to be present under %s.\n' \
                     'You can download it from https://stanfordnlp.github.io/CoreNLP/ and place it in this folder, or run: \n' \
                     '    bash get_stanford_models.sh\n' \
                     'If you prefer to skip SPICE, remove SPICE from config SCORER.TYPES or run with a smaller scorer set.\n' % cwd)

        # JVM options must precede -jar, otherwise Java treats them as jar names
        spice_cmd = ['java', '-Xmx8G', '-jar', SPICE_JAR, in_file.name,
          '-cache', self.cache_dir,
          '-out', out_file.name,
          '-subset',
          '-silent'
        ]
        try:
          subprocess.check_call(spice_cmd, 
           cwd=os.path.dirname(os.path.abspath(__file__)))
        except subprocess.CalledProcessError as e:
          # Provide a clearer error message for the user
          raise RuntimeError('SPICE scoring failed, please ensure Java is installed and the CoreNLP models jar is present in %s. Error: %s' % (cwd, e))
        except FileNotFoundError as e:
          raise RuntimeError('SPICE scoring failed because Java is not found in PATH. Please install Java (JRE/JDK) and ensure `java` is available in your PATH. Error: %s' % e)

        # Read and process results
        with open(out_file.name) as data_file:    
          results = json.load(data_file)
        os.remove(in_file.name)
        os.remove(out_file.name)

        spice_scores = []
        imgId_to_scores = {}
        for item in results:
          imgId_to_scores[item['image_id']] = item['scores']
          spice_scores.append(self.float_convert(item['scores']['All']['f']))
        average_score = np.mean(np.array(spice_scores))

        # Allow skipping per-image details for speed/memory when only the average is needed
        return_details = os.environ.get("SPICE_RETURN_DETAILS", "1").lower() not in ("0", "false", "no")
        if not return_details:
          return average_score, []

        scores = []
        for image_id in imgIds:
          # Convert none to NaN before saving scores over subcategories
          score_set = {}
          for category, score_tuple in imgId_to_scores[image_id].items():
            score_set[category] = {k: self.float_convert(v) for k, v in score_tuple.items()}
          scores.append(score_set)
        return average_score, scores

    def method(self):
        return "SPICE"

    def __del__(self):
        shutil.rmtree(self.cache_dir)
