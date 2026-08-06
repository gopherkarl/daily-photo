# Daily Photo Rotation

A local-first photography pipeline that selects one image from a personal photo pool, analyzes its visual composition, computes a phone-aware portrait crop position, updates a minimal GitHub Pages site, and publishes the result.

The project is optimized primarily for viewing on a portrait iPhone screen. Its central design goal is not merely to find the mathematical center of an image, but to preserve the photograph's meaningful subject when a landscape image must be displayed through a narrow portrait viewport.

Live site: [https://gopherkarl.github.io/daily-photo/](https://gopherkarl.github.io/daily-photo/)

## What the project does

Each rotation runs the following sequence:

1. Select the next unused photograph from the pool.
2. Copy it into the website as `photo.jpg`, converting HEIC/HEIF files to JPEG when necessary.
3. Calculate a quantitative visual-weight centroid from the image pixels.
4. Ask a local vision-language model to identify the primary subject and estimate its position in the frame.
5. Combine the pixel analysis, semantic subject analysis, and target-phone geometry to choose CSS `object-position` coordinates.
6. Update `index.html` with the selected crop position.
7. Save an auditable `analysis_report.json`.
8. Commit the changed site files and push them to GitHub Pages.

The pipeline runs locally through Ollama and does not require a cloud vision API for its automated image analysis.

## Pipeline architecture

```text
Photo pool
  |
  v
rotate.py
  - alphabetical rotation
  - cycle history
  - HEIC/HEIF conversion
  - photo.jpg + state.json
  |
  v
local_analyze.py
  |
  +--> Quantitative analysis
  |      - 100x100 pixel reduction
  |      - contrast, saturation, luminance deviation
  |      - weighted visual centroid
  |
  +--> Qualitative analysis
  |      - Qwen2.5-VL via Ollama
  |      - subject identification
  |      - estimated subject X/Y position
  |
  +--> Display-aware synthesis
         - Qwen2.5-Coder 32B via Ollama
         - portrait phone viewport geometry
         - final object-position coordinates
  |
  v
index.html + analysis_report.json
  |
  v
rotate_and_push.sh
  - git commit
  - git push origin main
```



## Repository files


| File                   | Purpose                                                                                       |
| ---------------------- | --------------------------------------------------------------------------------------------- |
| `index.html`           | Static responsive page displaying `photo.jpg` with `object-fit: cover`.                       |
| `photo.jpg`            | The currently selected and published photograph.                                              |
| `rotate.py`            | Selects the next image, copies/converts it, and updates rotation state.                       |
| `local_analyze.py` | Computes visual weight, requests structured subject localization, validates the crop, and updates the HTML. |
| `crop_geometry.py` | Calculates `object-fit: cover` geometry and checks subject bounding-box containment. |
| `display_profiles.json` | Configures portrait and landscape target viewport profiles. |
| `analysis_report.json` | Machine-readable record of the centroid, vision report, bounding box, crop positions, validation, and display context. |
| `qa_summary.md` | Human-readable crop QA report generated for each run. |
| `qa_source_preview.jpg` | Downscaled QA preview generated locally for inspection. |
| `state.json` | Tracks the last displayed photo and the current rotation history. |
| `rotate_and_push.sh` | Runs the complete locked analysis and publication sequence. |
| `tests/` | Automated tests for rotation and display geometry. |
| `.gitignore`           | Excludes local or generated files according to repository policy.                             |




## Photo selection

The source pool is configured in `rotate.py`:

```text
/Users/karl/Pictures/daily_photo_pool
```

Supported extensions are:

```text
.jpg .jpeg .png .webp .heic .heif .tiff
```

Selection is deterministic:

1. Files are sorted alphabetically.
2. Files already present in `state.json` history are excluded.
3. The first remaining file is selected.
4. When every file has been shown, the cycle resets while retaining the previous image temporarily to avoid an immediate repeat.

This means filenames can intentionally define the order, for example `01.jpg`, `02.jpg`, `03.jpg`.

## Quantitative visual-weight analysis

`local_analyze.py` reduces the image to a 100x100 grid and calculates three pixel-level signals:

- **Local contrast — 50%**: emphasizes edges and transitions, such as the boundary between a dark subject and a bright background.
- **Saturation — 30%**: emphasizes vivid and colorful regions.
- **Luminance deviation — 20%**: emphasizes pixels unusually bright or dark relative to the image.

The combined weight for each grid cell is:

```text
weight = 0.5 * contrast + 0.3 * saturation + 0.2 * luminance_deviation
```

The weighted center of mass of this map becomes the mathematical centroid:

```text
math_centroid = (x%, y%)
```

The centroid is a useful structural anchor, but it does not understand subjects or meaning. A bright window, colorful sign, high-contrast horizon, or reflection can attract the centroid even when the intended subject is elsewhere.

## Qualitative subject analysis

The current vision model is:

```text
Qwen2.5-VL 7B via Ollama: qwen2.5vl
```

The model receives a downscaled copy of the selected photo and must return valid JSON containing multiple semantic elements:

- `background_mass`: dense repetitive material such as a crowd, foliage, or architecture;
- `focal_anomaly`: an isolated person, animal, object, or pattern-breaker with high narrative value;
- `context`: field, signage, architecture, or other material needed to explain the scene;
- a complete bounding box for every element;
- a confidence score for every element.

This follows the skill’s crowd/high-frequency-texture rule: a large repetitive mass can dominate pixels without being the best crop target. The crop therefore prioritizes an informative anomaly while preserving at least one contextual element. For a soccer stadium, the crowd supplies scale and emotion, while the isolated player and field explain what the image is about.

If confidence is below the configured threshold, the pipeline performs a second independent vision pass and keeps the higher-confidence valid report. Invalid JSON, invalid coordinates, missing fields, or out-of-bounds boxes are rejected rather than silently used.

## Portrait-phone crop logic

The target displays are configured in `display_profiles.json`. The default profile is an iPhone 13 mini portrait viewport:

```text
375 x 812 CSS pixels
```

A second iPhone 13 mini landscape profile is also generated:

```text
812 x 375 CSS pixels
```

The dimensions are configuration, not code constants, so additional device profiles can be added without changing the geometry functions.

The website uses:

```css
.photo {
  width: 100%;
  height: 100%;
  object-fit: cover;
}
```



### Landscape source photo

A landscape image displayed in a portrait viewport is scaled until its full height fills the phone. Most of its width is then clipped. For a typical 3:2 landscape photo, approximately 30.7% of the source width is visible.

Therefore:

- `object-position-x` is critical;
- `object-position-y` has no visible effect because the full source height is already visible;
- the horizontal crop must contain the complete subject bounding box, not merely align with the mathematical centroid.

The crop position is calculated from the bounding box and the actual `object-fit: cover` overflow geometry. This is important because CSS `object-position: 80%` does not mean “put the subject at 80% of the phone screen.” It positions the overflow region. `crop_geometry.py` converts the subject box into the correct CSS coordinate and then verifies that the resulting visible source window contains the entire subject.

For a landscape source, the pipeline:

1. reads the validated focal-anomaly bounding box from the vision report;
2. computes the source region visible on the portrait target;
3. calculates the CSS X/Y position needed to center the anomaly where possible;
4. checks whether at least one context element is also visible;
5. preserves the primary anchor through an anchor-priority fallback when context cannot fit; it fails only if the primary anchor itself cannot be preserved. Other profiles remain reported for review.

This corrects the earlier point-estimate approximation and prevents a large repetitive crowd from displacing the informative subject. The resulting crop preserves both the anomaly and enough context to explain the scene.

### Portrait source photo

For a portrait source image, the current implementation uses the standard dual-core logic on both axes. Both X and Y may affect the visible crop, so the mathematical centroid and qualitative subject location can both contribute to the final position.

## Dual-core synthesis

The synthesis model is currently:

```text
Qwen3 32B via Ollama: qwen3:32b
```

Qwen2.5-VL performs visual extraction; Qwen3 performs text-based composition judgment over the structured element inventory and deterministic crop candidates. Qwen3 does not receive raw pixels in this stage.

It receives:

1. the quantitative centroid;
2. all detected elements and semantic roles;
3. candidate crop coordinates and visible source windows;
4. anchor, context, and anomaly containment/overlap results;
5. the target phone's display geometry.

The judge selects among candidates rather than inventing arbitrary coordinates. The governing hierarchy is:

```text
primary anchor > essential context > focal anomaly > secondary subject > background mass
```

The governing principle is:

> Preserve the element that gives the photograph its identity, retain enough context to explain it, and include an anomaly only when it strengthens the story without displacing the anchor.

The final report contains:

```json
{
  "math_centroid": {"x": 52, "y": 51},
  "visual_weight_analysis": "...",
  "final_x": 80,
  "final_y": 51,
  "justification": "...",
  "display_context": {
    "visible_fraction_x": 0.308,
    "visible_fraction_y": 1.0,
    "x_critical": true,
    "y_critical": false
  }
}
```

If the synthesis request fails, the code falls back to the mathematical centroid. This preserves pipeline continuity but may produce a less meaningful crop.

## Running locally



### Prerequisites

- macOS with `sips` available;
- Python 3;
- Python packages: `numpy`, `Pillow`, `torch`, `torchvision`, `transformers`;
- Ollama;
- the following Ollama models:

```bash
ollama pull qwen3-vl:8b
ollama pull qwen3:32b
```

The visual-language model proposes semantic labels, while `grounding_localizer.py`
uses the Hugging Face `IDEA-Research/grounding-dino-tiny` checkpoint to produce
independent bounding boxes. The default local environment is `.venv-grounding`;
the shell runner uses it automatically, or the interpreter can be overridden with
`DAILY_PHOTO_PYTHON`. Grounding DINO uses Apple MPS when available.

The repository currently uses absolute paths tailored to the original Mac installation. For reuse on another machine, update the paths in `rotate.py`, `local_analyze.py`, and the shell script, or refactor them into environment variables.

### Analyze the current `photo.jpg`

```bash
python3 local_analyze.py
```

This calculates the centroid, performs local vision analysis, calls the synthesis judge, writes `analysis_report.json`, and updates the crop style in `index.html`.

### Run the complete rotation and publication sequence

```bash
bash rotate_and_push.sh
```

This selects the next source image first, analyzes it, updates the site, commits the changes, and pushes to `origin/main`.

### Run selection only

```bash
python3 rotate.py
```

This changes `photo.jpg` and `state.json` but does not analyze or publish the image.

## Automation

The project is designed to run from an external scheduler. The current Hermes wrapper is:

```text
/Users/karl/.hermes/scripts/daily_photo_run.sh
```

The wrapper invokes `rotate_and_push.sh`, captures output, remains silent on success, and reports output on failure. The configured scheduled job runs daily at 6:30 AM.

Schedulers should invoke the wrapper rather than calling `rotate_and_push.sh` through an incorrect relative path. The wrapper exists because scheduler script paths resolve relative to the scheduler's script directory rather than the repository directory.

## GitHub Pages publication

The repository's `main` branch is the publication source. The shell pipeline stages the generated site artifacts, creates a rotation commit, and pushes to GitHub:

```bash
git add photo.jpg state.json index.html local_analyze.py rotate_and_push.sh analysis_report.json
git commit -m "Rotate to photo: <filename>"
git push origin main
```

GitHub Pages then serves the static site at the configured project URL.

## Known limitations

- **Subject labels and coordinates come from separate models.** Qwen3-VL proposes semantic elements; Grounding DINO-T localizes them. Detector confidence and crop containment are still not proof of artistic quality.
- **Local vision can still be wrong.** The pipeline now uses scene consistency checks, semantic roles, candidate scoring, and validation, but a visually incorrect inventory can still require manual review.
- **The target viewport is hard-coded.** The primary geometry is based on an iPhone 13 mini in portrait mode. Other phones, browser chrome, safe-area behavior, or landscape viewing can produce different visible windows.
- **Only one CSS crop is published.** The page does not currently use separate portrait and landscape crop positions via media queries.
- **The HTML patch is regex-based.** `local_analyze.py` expects an inline `object-position` style attribute in `index.html`.
- **Invalid portrait composition now fails the run.** The scheduled wrapper therefore does not publish a crop lacking both an inside anchor and explicit context.
- **The pipeline assumes a clean, serialized run.** Concurrent executions could race while modifying `photo.jpg`, `state.json`, `index.html`, or `analysis_report.json`.
- **Generated files are committed.** This makes each published crop auditable, but it also causes large image files and analysis reports to accumulate in Git history.



## Future improvements

The following improvements are now implemented:

- structured JSON output from Qwen2.5-VL through the Ollama HTTP API;
- multi-element semantic roles (`primary_anchor`, `background_mass`, `focal_anomaly`, `context`);
- subject bounding boxes, confidence scores, and narrative values for every element;
- scene-consistency verification and second-pass analysis when context is missing or confidence is low;
- deterministic candidate crop generation and scoring;
- Qwen3:32B composition judgment over candidate crops;
- validation that the primary anchor and meaningful context are visible;
- overlap-based handling for distributed masses such as crowds;
- separate portrait and landscape crop positions with CSS media queries;
- configurable display profiles;
- a lock directory to prevent concurrent rotations;
- atomic `state.json` writes;
- input, model-output, HTML, and crop-validation checks;
- automated tests for rotation cycles and crop geometry;
- a deterministic `qa_summary.md` audit artifact and downscaled `qa_source_preview.jpg`.

Remaining improvements include:

- use a dedicated visual QA renderer to produce exact portrait and landscape screenshots;
- add a second judge model or human-review mode when bounding-box validation fails;
- separate generated analysis artifacts from the production branch if Git history becomes too large;
- consider Git LFS or an image archive as the photo collection grows.



## Design philosophy

The project combines two imperfect forms of knowledge:

- **Pixel statistics** provide a reproducible account of contrast, color, and luminance distribution.
- **Vision-language reasoning** provides a semantic account of what the photograph is about and where its subject is located.

Neither is sufficient alone. The mathematical centroid is blind to meaning; the vision model can be vague or mistaken. The useful result comes from exposing both sources of error, checking them against the physical constraints of the target display, and producing an auditable crop decision rather than pretending that a single heuristic is universally correct.

## License and image rights

The code and the photographs may have different rights. Before publishing this repository publicly, add an explicit license for the code and confirm that every image in the repository may be redistributed through GitHub and GitHub Pages.

## Status

This is a personal, local-first project under active development. The current implementation is optimized for portrait-phone viewing and uses Qwen2.5-VL for local subject localization and Qwen2.5-Coder for crop synthesis.

*Last updated: 2026-08-04*