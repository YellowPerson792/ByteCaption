#!/usr/bin/env python

# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import subprocess
import threading
import urllib.request

# Assumes meteor-1.5.jar is in the same directory as meteor.py.  Change as needed.
METEOR_JAR = 'meteor-1.5.jar'
# print METEOR_JAR

class Meteor:

    def __init__(self):
        self.env = os.environ
        self.env['LC_ALL'] = 'en_US.UTF_8'
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        para_path = os.path.join(data_dir, 'paraphrase-en.gz')
        if not os.path.exists(para_path):
            os.makedirs(data_dir, exist_ok=True)
            try:
                # Fetch the missing paraphrase dictionary required by METEOR
                url = 'https://raw.githubusercontent.com/tylin/coco-caption/master/pycocoevalcap/meteor/data/paraphrase-en.gz'
                urllib.request.urlretrieve(url, para_path)
            except Exception as e:
                raise RuntimeError(
                    'METEOR requires paraphrase-en.gz under %s. Auto-download failed: %s' % (data_dir, e)
                )
        # JVM options (e.g., -Xmx) must precede the -jar flag; the previous
        # order would cause Java to treat -Xmx as the jar name and exit
        # immediately, leading to BrokenPipeError in compute_score.
        self.meteor_cmd = ['java', '-Xmx2G', '-jar', METEOR_JAR,
                '-', '-', '-stdio', '-l', 'en', '-norm']
        self.meteor_p = subprocess.Popen(
                self.meteor_cmd,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.env,
                universal_newlines=True,
                encoding="utf-8",
                bufsize=1
        )
        # Used to guarantee thread safety
        self.lock = threading.Lock()

    def compute_score(self, gts, res):
        assert(gts.keys() == res.keys())
        imgIds = sorted(list(gts.keys()))
        scores = []

        eval_line = 'EVAL'
        self.lock.acquire()
        for i in imgIds:
            assert(len(res[i]) == 1)
            stat = self._stat(res[i][0], gts[i])
            eval_line += ' ||| {}'.format(stat)

        # Send to METEOR
        self.meteor_p.stdin.write(eval_line + '\n')
        
        # Collect segment scores
        for i in range(len(imgIds)):
            score = float(self.meteor_p.stdout.readline().strip())
            scores.append(score)

        # Final score
        final_score = float(self.meteor_p.stdout.readline().strip())
        self.lock.release()

        return final_score, scores

    def method(self):
        return "METEOR"

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write(score_line+'\n')
        return self.meteor_p.stdout.readline().strip()
 
    def __del__(self):
        """Best-effort cleanup without blocking interpreter shutdown.

        The original implementation could hang at program exit because it
        always tried to acquire the lock and wait on the Java process even
        when Python was already shutting down (e.g. after Ctrl+C or during
        GC of globals). Here we:
        - Guard against missing attributes during interpreter teardown.
        - Use a non-blocking acquire so that we never deadlock on exit.
        - Swallow all exceptions, since this runs in garbage collection
          context where reliability is more important than strict cleanup.
        """
        try:
            lock = getattr(self, "lock", None)
            proc = getattr(self, "meteor_p", None)
            if lock is None or proc is None:
                return

            # Try non-blocking acquire; if we can't get it immediately,
            # just give up to avoid hanging at shutdown.
            acquired = lock.acquire(blocking=False)
            if not acquired:
                return

            try:
                # stdin may already be closed; ignore any errors.
                try:
                    if proc.stdin and not proc.stdin.closed:
                        proc.stdin.close()
                except Exception:
                    pass

                try:
                    proc.kill()
                except Exception:
                    pass

                try:
                    # Use small timeout to avoid blocking indefinitely.
                    proc.wait(timeout=1.0)
                except Exception:
                    pass
            finally:
                lock.release()
        except Exception:
            # Never raise from __del__
            pass
