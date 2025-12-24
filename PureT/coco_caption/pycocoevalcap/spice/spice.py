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
import atexit

import time
import shutil

# Assumes spice.jar is in the same directory as spice.py.  Change as needed.
SPICE_JAR = 'spice-1.0.jar'
TEMP_DIR = 'tmp'
CACHE_DIR = 'cache'
SERVER_CLASSES_DIR = 'server_classes'
SERVER_SRC = os.path.join('server', 'edu', 'anu', 'spice', 'SpiceServer.java')
SERVER_CLASS = os.path.join('edu', 'anu', 'spice', 'SpiceServer.class')

_SPICE_SERVER = None
_SPICE_SERVER_LOCK = threading.Lock()

_JAVA_MAJOR_VERSION = None


def _java_major_version():
  """Best-effort detection of Java major version.

  Returns an int (e.g. 8, 11, 17, 21) or None if undetectable.
  """
  global _JAVA_MAJOR_VERSION
  if _JAVA_MAJOR_VERSION is not None:
    return _JAVA_MAJOR_VERSION
  try:
    # java -version prints to stderr
    proc = subprocess.run(
      ["java", "-version"],
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      universal_newlines=True,
      check=False,
    )
    text = (proc.stderr or "") + "\n" + (proc.stdout or "")
    first_line = (text.strip().splitlines() or [""])[0]
    # Examples:
    # - openjdk version "17.0.10" 2024-01-16
    # - java version "1.8.0_392"
    import re

    m = re.search(r'version\s+"([0-9]+)(?:\.([0-9]+))?', first_line)
    if not m:
      _JAVA_MAJOR_VERSION = None
      return _JAVA_MAJOR_VERSION
    major = int(m.group(1))
    minor = m.group(2)
    # Java 8 reports as 1.8
    if major == 1 and minor is not None:
      major = int(minor)
    _JAVA_MAJOR_VERSION = major
    return _JAVA_MAJOR_VERSION
  except Exception:
    _JAVA_MAJOR_VERSION = None
    return _JAVA_MAJOR_VERSION


def _spice_jvm_args():
  """Return JVM args needed for SPICE on newer Java runtimes.

  Java 16+ tightens reflection access; SPICE's dependencies may require
  opening java.base internals (commonly fails on Windows where newer JDKs are installed).
  We only add these flags when Java >= 16 to remain compatible with Java 8.
  """
  # Allow user override/extension
  extra = os.environ.get("SPICE_JAVA_OPTS", "").strip()
  extra_args = extra.split() if extra else []

  major = _java_major_version()
  if major is None or major < 16:
    return extra_args

  add_opens = [
    "--add-opens", "java.base/java.lang=ALL-UNNAMED",
    "--add-opens", "java.base/java.util=ALL-UNNAMED",
    "--add-opens", "java.base/java.util.concurrent=ALL-UNNAMED",
    "--add-opens", "java.base/java.io=ALL-UNNAMED",
    "--add-opens", "java.base/java.math=ALL-UNNAMED",
    "--add-opens", "java.base/java.nio=ALL-UNNAMED",
    "--add-opens", "java.base/java.net=ALL-UNNAMED",
    "--add-opens", "java.base/java.text=ALL-UNNAMED",
    "--add-opens", "java.base/java.time=ALL-UNNAMED",
    "--add-opens", "java.base/sun.nio.ch=ALL-UNNAMED",
  ]
  return add_opens + extra_args


def _env_flag(name):
    value = os.environ.get(name, "").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    return None


def _server_classpath(cwd):
    server_classes = os.path.join(cwd, SERVER_CLASSES_DIR)
    server_class = os.path.join(server_classes, SERVER_CLASS)
    if not os.path.exists(server_class):
        return None
    spice_jar = os.path.join(cwd, SPICE_JAR)
    lib_glob = os.path.join(cwd, 'lib', '*')
    return os.pathsep.join([server_classes, spice_jar, lib_glob])


def _ensure_server_compiled(cwd):
    server_classes = os.path.join(cwd, SERVER_CLASSES_DIR)
    server_class = os.path.join(server_classes, SERVER_CLASS)
    if os.path.exists(server_class):
        return True
    src_path = os.path.join(cwd, SERVER_SRC)
    if not os.path.exists(src_path):
        return False
    javac = shutil.which("javac")
    if not javac:
        return False
    os.makedirs(server_classes, exist_ok=True)
    classpath = os.pathsep.join([
        os.path.join(cwd, SPICE_JAR),
        os.path.join(cwd, 'lib', '*'),
    ])
    try:
        subprocess.check_call(
            [javac, '-cp', classpath, '-d', server_classes, src_path],
            cwd=cwd
        )
    except Exception:
        return False
    return os.path.exists(server_class)


class _SpiceServerProcess:
    def __init__(self, cwd, cache_dir, threads=None, use_synsets=True):
        self._lock = threading.Lock()
        self._next_id = 0
        self.cache_dir = cache_dir
        classpath = _server_classpath(cwd)
        if not classpath:
            raise RuntimeError("SPICE server classes not available")
        cmd = ['java'] + _spice_jvm_args() + ['-Xmx8G', '-cp', classpath, 'edu.anu.spice.SpiceServer']
        cmd += ['-cache', cache_dir, '-subset', '-silent']
        if threads:
            cmd += ['-threads', str(threads)]
        if not use_synsets:
            cmd += ['-noSynsets']
        self.proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _drain_stderr(self):
        if not self.proc or not self.proc.stderr:
            return
        for line in self.proc.stderr:
            sys.stderr.write(line)

    def score(self, input_path, output_path):
        if self.proc.poll() is not None:
            raise RuntimeError("SPICE server is not running")
        with self._lock:
            self._next_id += 1
            payload = {"id": self._next_id, "input": input_path, "output": output_path}
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("SPICE server returned no response")
            try:
                resp = json.loads(line)
            except Exception as exc:
                raise RuntimeError("SPICE server response parse failed: %s" % exc)
            if not resp.get("ok", False):
                raise RuntimeError(resp.get("error", "SPICE server error"))

    def close(self):
        if not self.proc:
            return
        try:
            if self.proc.poll() is None and self.proc.stdin:
                self.proc.stdin.write("exit\n")
                self.proc.stdin.flush()
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass
        self.proc = None


def _close_spice_server():
    global _SPICE_SERVER
    if _SPICE_SERVER:
        _SPICE_SERVER.close()
        _SPICE_SERVER = None


atexit.register(_close_spice_server)


def _safe_unlink(path, retries=5, delay=0.1):
    """Best-effort file removal (Windows can hold temp files briefly)."""
    if not path:
        return
    for attempt in range(retries + 1):
        try:
            os.remove(path)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt >= retries:
                return
            time.sleep(delay)
        except Exception:
            return


def _get_spice_server(cwd, cache_dir):
    global _SPICE_SERVER
    use_server = _env_flag("SPICE_SERVER")
    if use_server is False:
        return None
    if _server_classpath(cwd) is None and use_server is True:
        _ensure_server_compiled(cwd)
    if _server_classpath(cwd) is None:
        return None
    threads_env = os.environ.get("SPICE_THREADS", "").strip()
    threads = int(threads_env) if threads_env.isdigit() else None
    use_synsets = _env_flag("SPICE_NO_SYNSETS") is not True
    with _SPICE_SERVER_LOCK:
        if _SPICE_SERVER and _SPICE_SERVER.cache_dir != cache_dir:
            _SPICE_SERVER.close()
            _SPICE_SERVER = None
        if _SPICE_SERVER is None:
            _SPICE_SERVER = _SpiceServerProcess(
                cwd,
                cache_dir,
                threads=threads,
                use_synsets=use_synsets
            )
        return _SPICE_SERVER

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
        self._cleanup_cache = os.environ.get("SPICE_CLEANUP_CACHE", "").strip().lower() in ("1", "true", "yes")
        self._ensure_cache_dir()

    def _ensure_cache_dir(self):
        """Ensure cache directory exists and is not pruned as empty."""
        if not os.path.exists(self.cache_dir):
          os.makedirs(self.cache_dir, exist_ok=True)
        keep_path = os.path.join(self.cache_dir, ".keep")
        try:
          with open(keep_path, "a"):
            pass
        except Exception:
          pass

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

        server_used = False
        server = _get_spice_server(cwd, self.cache_dir)
        if server:
          try:
            server.score(in_file.name, out_file.name)
            server_used = True
          except Exception as e:
            sys.stderr.write("SPICE server failed, falling back to jar: %s\n" % e)
            _close_spice_server()

        if not server_used:
          # JVM options must precede -jar, otherwise Java treats them as jar names
          spice_cmd = ['java'] + _spice_jvm_args() + ['-Xmx8G', '-jar', SPICE_JAR, in_file.name,
            '-cache', self.cache_dir,
            '-out', out_file.name,
            '-subset',
            '-silent'
          ]
          threads_env = os.environ.get("SPICE_THREADS", "").strip()
          if threads_env.isdigit():
            spice_cmd += ['-threads', threads_env]
          if _env_flag("SPICE_NO_SYNSETS") is True:
            spice_cmd += ['-noSynsets']
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
        _safe_unlink(in_file.name)
        _safe_unlink(out_file.name)
        # Some environments prune empty cache directories; keep it persistent.
        self._ensure_cache_dir()

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
        if self._cleanup_cache and os.path.isdir(self.cache_dir):
          shutil.rmtree(self.cache_dir, ignore_errors=True)
