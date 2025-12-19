/*
 * Persistent SPICE server that keeps a single Stanford CoreNLP pipeline
 * alive across multiple requests to avoid re-initialization overhead.
 */
package edu.anu.spice;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import com.google.common.base.Stopwatch;

public class SpiceServer {

    private final SpiceParser parser;
    private final boolean useSynsets;
    private final boolean detailed;
    private final boolean tupleSubsets;
    private final boolean silent;

    public SpiceServer(String cacheDir,
                       int numThreads,
                       boolean useSynsets,
                       boolean tupleSubsets,
                       boolean detailed,
                       boolean silent) {
        this.useSynsets = useSynsets;
        this.detailed = detailed;
        this.tupleSubsets = tupleSubsets;
        this.silent = silent;
        this.parser = new SpiceParser(cacheDir, numThreads, useSynsets);
    }

    public static void main(String[] args) {
        // Server-mode args (similar to SpiceArguments options, without input/output path)
        String cache = null;
        int numThreads = Runtime.getRuntime().availableProcessors();
        boolean detailed = false;
        boolean synsets = true;
        boolean tupleSubsets = false;
        boolean silent = false;

        int curArg = 0;
        while (curArg < args.length) {
            if (args[curArg].equals("-cache")) {
                cache = args[curArg + 1];
                curArg += 2;
            } else if (args[curArg].equals("-threads")) {
                numThreads = Integer.parseInt(args[curArg + 1]);
                curArg += 2;
            } else if (args[curArg].equals("-detailed")) {
                detailed = true;
                curArg += 1;
            } else if (args[curArg].equals("-noSynsets")) {
                synsets = false;
                curArg += 1;
            } else if (args[curArg].equals("-subset")) {
                tupleSubsets = true;
                curArg += 1;
            } else if (args[curArg].equals("-silent")) {
                silent = true;
                curArg += 1;
            } else {
                System.err.println("Unknown option \"" + args[curArg] + "\"");
                System.exit(1);
            }
        }

        SpiceServer server = new SpiceServer(cache, numThreads, synsets, tupleSubsets, detailed, silent);
        try {
            server.serve();
        } catch (Exception ex) {
            System.err.println("Error: SpiceServer failed.");
            ex.printStackTrace();
            System.exit(1);
        }
    }

    private Map<String, TupleFilter> buildFilters(boolean tupleSubsets) {
        Map<String, TupleFilter> filters = new HashMap<String, TupleFilter>();
        if (tupleSubsets) {
            filters.put("Object", TupleFilter.objectFilter);
            filters.put("Attribute", TupleFilter.attributeFilter);
            filters.put("Relation", TupleFilter.relationFilter);
            filters.put("Cardinality", TupleFilter.cardinalityFilter);
            filters.put("Color", TupleFilter.colorFilter);
            filters.put("Size", TupleFilter.sizeFilter);
        }
        return filters;
    }

    private void serve() throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in, "UTF-8"));
        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(System.out, "UTF-8"));
        JSONParser json = new JSONParser();
        String line;
        while ((line = reader.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty()) {
                continue;
            }
            if (line.equalsIgnoreCase("quit") || line.equalsIgnoreCase("exit")) {
                writer.write("{\"ok\":true,\"exit\":true}\n");
                writer.flush();
                break;
            }

            Object reqId = null;
            String inputPath = null;
            String outputPath = null;
            try {
                JSONObject req = (JSONObject) json.parse(line);
                reqId = req.get("id");
                inputPath = (String) req.get("input");
                outputPath = (String) req.get("output");
            } catch (ParseException e) {
                writeError(writer, reqId, "Invalid JSON request: " + e.toString());
                continue;
            } catch (ClassCastException e) {
                writeError(writer, reqId, "Invalid request format.");
                continue;
            }

            if (inputPath == null || outputPath == null) {
                writeError(writer, reqId, "Request requires \"input\" and \"output\".");
                continue;
            }

            try {
                scoreOnce(inputPath, outputPath);
                writeOk(writer, reqId);
            } catch (Exception ex) {
                writeError(writer, reqId, ex.toString());
            }
        }
    }

    private void writeOk(BufferedWriter writer, Object reqId) throws IOException {
        JSONObject resp = new JSONObject();
        resp.put("ok", true);
        if (reqId != null) {
            resp.put("id", reqId);
        }
        writer.write(resp.toJSONString());
        writer.write("\n");
        writer.flush();
    }

    private void writeError(BufferedWriter writer, Object reqId, String error) throws IOException {
        JSONObject resp = new JSONObject();
        resp.put("ok", false);
        resp.put("error", error);
        if (reqId != null) {
            resp.put("id", reqId);
        }
        writer.write(resp.toJSONString());
        writer.write("\n");
        writer.flush();
    }

    private void scoreOnce(String inputPath, String outputPath) throws IOException, ParseException {
        Stopwatch timer = Stopwatch.createStarted();
        Map<String, TupleFilter> filters = buildFilters(this.tupleSubsets);

        ArrayList<Object> imageIds = new ArrayList<Object>();
        ArrayList<String> testCaptions = new ArrayList<String>();
        ArrayList<Integer> testChunks = new ArrayList<Integer>();
        ArrayList<String> refCaptions = new ArrayList<String>();
        ArrayList<Integer> refChunks = new ArrayList<Integer>();
        JSONParser json = new JSONParser();
        JSONArray input = (JSONArray) json.parse(new FileReader(inputPath));
        for (Object o : input) {
            JSONObject item = (JSONObject) o;
            imageIds.add(item.get("image_id"));
            JSONArray tests = (JSONArray) item.get("tests");
            testChunks.add(tests.size());
            for (Object test : tests) {
                testCaptions.add((String) test);
            }
            JSONArray refs = (JSONArray) item.get("refs");
            refChunks.add(refs.size());
            for (Object ref : refs) {
                refCaptions.add((String) ref);
            }
        }

        System.err.println("Parsing reference captions");
        List<SceneGraph> refSgs = parser.parseCaptions(refCaptions, refChunks);
        System.err.println("Parsing test captions");
        List<SceneGraph> testSgs = parser.parseCaptions(testCaptions, testChunks);

        SpiceStats stats = new SpiceStats(filters, this.detailed);
        for (int i = 0; i < testSgs.size(); ++i) {
            stats.score(imageIds.get(i), testSgs.get(i), refSgs.get(i), this.useSynsets);
        }
        if (!this.silent) {
            System.out.println(stats.toString());
        }

        BufferedWriter outputWriter = new BufferedWriter(new FileWriter(outputPath));
        outputWriter.write(stats.toJSONString());
        outputWriter.close();
        System.err.println("SPICE evaluation took: " + timer.stop());
    }
}
