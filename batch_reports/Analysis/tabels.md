<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="6">RBBF</th>
    </tr>
    <tr>
      <th>S0</th>
      <th>S1</th>
      <th>S2</th>
      <th>S3</th>
      <th>S4</th>
      <th>S5</th>
    </tr>
  </thead>
  <tbody>
  <tr><th colspan="7">COCO Pretrained Models</th></tr>
  <tr><td>BLIP</td><td>121.5 / 21.2</td><td>45.6 / 9.8</td><td>8.4 / 2.9</td><td>5.4 / 2.6</td><td>3.2 / 1.5</td><td>1.3 / 0.6</td></tr>
  <tr><td>GIT</td><td><strong>135.1</strong> / 23.3</td><td>51.4 / 9.8</td><td>4.4 / 1.9</td><td>2.0 / 1.3</td><td>1.1 / 0.6</td><td>1.3 / 0.6</td></tr>
  <tr><th colspan="7">COCO Fine-Tuned Models</th></tr>
  <tr><td>Qwen3-VL-8B</td><td>127.0 / <strong>23.5</strong></td><td><strong>82.0</strong> / 15.8</td><td>21.7 / 5.5</td><td>4.5 / 2.0</td><td>2.4 / 1.2</td><td>1.0 / 0.7</td></tr>
  <tr><th colspan="7">Zero-Shot Generative Models</th></tr>
  <tr><td>GPT-5.1</td><td>77.9 / 19.1</td><td>55.5 / 14.0</td><td>16.7 / 5.1</td><td>3.0 / 1.4</td><td>1.0 / 0.8</td><td>1.0 / 0.7</td></tr>
  <tr><td>Gemini-2.5-flash</td><td>119.4 / 23.4</td><td>81.1 / <strong>16.6</strong></td><td>18.5 / 4.9</td><td>3.0 / 1.4</td><td>1.1 / 0.9</td><td>1.0 / 0.7</td></tr>
  <tr><td>Claude-Haiku-4.5</td><td>55.5 / 23.4</td><td>29.5 / 8.3</td><td>3.5 / 1.5</td><td>1.0 / 0.7</td><td>1.0 / 0.7</td><td>1.0 / 0.7</td></tr>
  <tr><th colspan="7">Ours</th></tr>
  <tr><td>ByteCaption</td><td>66.4 / 13.4</td><td>66.3 / 13.4</td><td><strong>64.9</strong> / <strong>13.0</strong></td><td><strong>63.4</strong> / <strong>12.6</strong></td><td><strong>54.6</strong> / <strong>11.2</strong></td><td><strong>7.1</strong> / <strong>3.1</strong></td></tr>
  </tbody>
</table>

## Visualizations

### Curves (CIDEr/SPICE)

![RBBF curves](curves_rbbf.png)

![RBSL curves](curves_rbsl.png)

### Drop heatmaps (relative to S0)

![RBBF drop heatmap](heatmap_drop_rbbf.png)

![RBSL drop heatmap](heatmap_drop_rbsl.png)

### Aggregate robustness score

![Robustness score](robustness_score.png)

Summary CSV: [robustness_summary.csv](robustness_summary.csv)

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="6">RBSL</th>
    </tr>
    <tr>
      <th>S0</th>
      <th>S1</th>
      <th>S2</th>
      <th>S3</th>
      <th>S4</th>
      <th>S5</th>
    </tr>
  </thead>
  <tbody>
  <tr><th colspan="7">COCO Pretrained Models</th></tr>
  <tr><td>BLIP</td><td>121.5 / 21.2</td><td>37.8 / 8.3</td><td>15.8 / 4.6</td><td>4.6 / 1.4</td><td>1.9 / 0.8</td><td>1.4 / 0.6</td></tr>
  <tr><td>GIT</td><td><strong>135.1</strong> / 23.3</td><td>40.6 / 8.4</td><td>12.8 / 3.7</td><td>4.0 / 1.2</td><td>1.3 / 0.7</td><td>1.4 / 0.7</td></tr>
  <tr><th colspan="7">COCO Fine-Tuned Models</th></tr>
  <tr><td>Qwen3-VL-8B</td><td>127.0 / <strong>23.5</strong></td><td>51.9 / 11.0</td><td>31.9 / 7.2</td><td>7.5 / 2.4</td><td>1.3 / 0.8</td><td>1.0 / 0.8</td></tr>
  <tr><th colspan="7">Zero-Shot Generative Models</th></tr>
  <tr><td>GPT-5.1</td><td>77.9 / 19.1</td><td>36.9 / 10.2</td><td>25.3 / 7.4</td><td>6.8 / 2.5</td><td>1.1 / 0.8</td><td>1.0 / 0.7</td></tr>
  <tr><td>Gemini-2.5-flash</td><td>119.4 / 23.4</td><td>52.8 / 11.8</td><td>31.8 / 7.5</td><td>6.9 / 2.3</td><td>1.2 / 0.8</td><td>1.0 / 0.8</td></tr>
  <tr><td>Claude-Haiku-4.5</td><td>55.5 / 23.4</td><td>17.5 / 5.5</td><td>10.2 / 3.5</td><td>2.5 / 1.2</td><td>1.0 / 0.8</td><td>1.0 / 0.7</td></tr>
  <tr><th colspan="7">Ours</th></tr>
  <tr><td>ByteCaption</td><td>66.4 / 13.4</td><td><strong>65.8</strong> / <strong>13.6</strong></td><td><strong>65.9</strong> / <strong>13.2</strong></td><td><strong>63.9</strong> / <strong>13.1</strong></td><td><strong>50.6</strong> / <strong>10.6</strong></td><td><strong>9.3</strong> / <strong>2.7</strong></td></tr>
  </tbody>
</table>
