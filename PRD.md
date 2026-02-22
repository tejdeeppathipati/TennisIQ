# TennisIQ — Product Requirements Document

**Tennis Vision Intelligence for College Coaches**

---

## Section 1: Premise

### 1.1 The Problem

College tennis coaches have no affordable, automated way to extract meaningful insights from match footage. Analyzing a single match requires manually scrubbing through hours of video, mentally tracking serve placement, rally patterns, player positioning, and error tendencies — a process that takes days and relies entirely on human memory and attention.

Professional tracking systems like Hawkeye and PlaySight exist but require proprietary hardware installations costing tens of thousands of dollars — completely inaccessible to college programs. The result is that most college coaching staffs are making strategic decisions on incomplete, manually processed information while their players develop slower than they could with proper analytics feedback.

### 1.2 The Problem Quantified

A college tennis coaching staff reviews approximately 2–4 hours of match footage per opponent per week across singles and doubles matchups. With a typical 25-match college season, that is roughly 50–100 hours of raw footage reviewed manually per season. Research on sports analytics workflows suggests coaches spend up to 60% of film review time on mechanical tasks — scrubbing, rewinding, clipping — rather than actual strategic analysis.

That is 30–60 hours per season lost to tasks a machine could handle. The gap between what well-funded professional programs can see and what college programs can afford to see is enormous and growing.

### 1.3 The Gap

Existing solutions fail on one of three dimensions. Generic CV tools require significant ML expertise and produce no tennis-specific insights. Professional tracking systems require proprietary hardware. Open-source pipelines produce bounding boxes and weights but no coaching intelligence — no serve analysis, no rally tracking, no court coverage heatmaps, no point-by-point coaching cards.

Nobody has built a system that takes a YouTube link or court camera recording of a college tennis match and produces accurate serve placement charts, error heatmaps, point-by-point coaching cards, and side-by-side annotated video — automatically, affordably, and without any ML expertise from the coach.

### 1.4 The Solution

TennisIQ is an end-to-end tennis vision intelligence platform that takes raw college tennis footage and produces a complete coaching analytics package in minutes — not days. It uses two purpose-built pretrained computer vision models specifically trained on tennis match data to extract ball trajectory, court geometry, and player positions from every frame of the uploaded video. These detections feed into a physics-based analytics engine that identifies every point in the match, classifies serve placement, detects bounces and hits, determines in/out calls using real court geometry, and generates per-point coaching cards explaining what happened and what to try differently.

The coach uploads footage, reviews the system's detections at a checkpoint, flags anything that looks wrong, and receives a complete analytics dashboard with side-by-side sync video, serve placement charts, error heatmaps, player movement heatmaps, and match-level strategic insights.

No ML expertise required. No hardware installation. No training wait.

### 1.5 Why Pretrained Tennis-Specific Models

TennisIQ is built on purpose-trained models for tennis vision, not generic object detectors:

**ResNet50 Court Keypoint Regression** is a convolutional network trained on annotated tennis court images. It detects 14 precise court line intersection points per frame — baseline corners, service line corners, net posts, center marks — by directly regressing 28 coordinate values (14 x,y pairs) from a 224x224 input. These 14 keypoints are used to compute a homography matrix every frame, transforming any pixel coordinate in the camera view into a real coordinate on a standardized top-down court diagram. This is what makes every downstream analytic — serve placement, in/out classification, player heatmaps — mathematically accurate rather than pixel approximations.

**YOLOv5 Tennis Ball Detector** is a YOLOv5L6u model fine-tuned specifically on tennis footage to detect the ball via bounding box regression. A tennis ball at match speed is often just a few pixels wide and motion-blurred, but this model was trained exclusively on tennis match frames and handles these conditions reliably. Each frame produces a bounding box with confidence, from which the ball center is extracted.

**YOLOv8n with ByteTrack** handles player detection as a general person detector, constrained by court boundary filtering using the homography to eliminate ball boys, line judges, chair umpires, and crowd members from the valid player set. If YOLOv8n is unavailable, the system falls back to OpenCV HOG-based person detection automatically.

These three models together give TennisIQ real court-space coordinates for every ball position, player position, and bounce event in the match — the same geometric foundation that professional systems like Hawkeye use, without the proprietary hardware.

### 1.6 The Homography — Why It Matters

The homography matrix is the technical foundation of everything TennisIQ produces. Every frame, the ResNet50 court keypoint model detects 14 court keypoints. Those 14 points are used to compute a perspective transformation matrix that maps pixel coordinates to real court coordinates.

**Without homography:**
- Ball at pixel (450, 312) → meaningless number
- "Serve landed near the right side" → vague guess

**With homography:**
- Ball at pixel (450, 312) → court coordinate (950, 1200)
- Court coordinate (950, 1200) → inside top-right service box → IN
- Court coordinate (1450, 900) → outside singles line → FAULT WIDE

This is what separates TennisIQ from a system that draws boxes on video. Every serve placement call, every in/out classification, every error heatmap point, every player movement heatmap is grounded in real court geometry — the same mathematics Hawkeye uses, without the hardware.

### 1.7 Features

**Inference-Only Pipeline.** No training on the coach's footage. The pretrained models run on uploaded video immediately, producing results in minutes rather than 30+ minutes.

**Accurate Ball Tracking.** YOLOv5 detects the ball via bounding box on every visible frame, handling motion blur and small ball sizes. Gaps are filled via interpolation, and outliers are removed via distance thresholding.

**Court Keypoint Detection and Homography.** ResNet50 court keypoint regression detects 14 court keypoints per frame. Homography computed every frame with temporal stabilization — if keypoints are briefly occluded, the last reliable homography is carried forward for up to 5 frames.

**Player Detection with Court Boundary Filtering.** YOLOv8n detects persons in every frame (with HOG fallback). Court boundary filtering using the homography discards any detection whose foot position projects outside the valid court region, eliminating ball boys, line judges, umpires, and crowd members. Carry-forward logic maintains player positions on zero-detection frames.

**Physics-Based Event Detection.** Bounce detection identifies frames where ball vertical velocity reverses and speed drops — the physical signature of a bounce. Hit detection identifies frames where ball direction changes 50+ degrees. Both are scored using heuristic kinematic features normalized to confidence scores, with temporal NMS to remove duplicates.

**Point Segmentation.** A state machine segments continuous match video into discrete tennis points — IDLE → SERVE → RALLY → IDLE — with end reason classification: OUT, FAULT, DOUBLE BOUNCE, NET, BALL LOST.

**In/Out Classification.** Every bounce is classified using real court polygon geometry in court-space coordinates. Not pixel guessing — actual geometric classification against the singles polygon, with a line margin for calls on or near the line.

**Serve Analysis.** First bounce per point classified into service box (top-left, top-right, bottom-left, bottom-right). Fault type classified as wide, long, or net. Serve in percentage computed per match.

**Coach Checkpoint Review.** After inference completes, coach reviews detected points — confirming, flagging, or noting corrections on tennis calls (not bounding boxes). Feedback saved to session for future matches.

**Side-by-Side Sync Video Player.** Raw footage left, TennisIQ-analyzed overlay right, synced on a single scrub bar. Clicking any point in the timeline jumps both videos simultaneously to that moment.

**Per-Point Coaching Cards.** For every detected point, a plain-English card explains why the point ended and what to try differently. Grounded in actual detection data — serve zone, end reason, rally length, court position.

**Court Coverage Heatmaps.** Player movement aggregated in real court coordinates showing where each player spent time. Gaps in coverage visible immediately.

**Error Heatmap.** Every out bounce plotted in court-space coordinates showing where errors cluster — deuce side, ad side, baseline, net.

**Serve Placement Chart.** Top-down court diagram with every serve plotted — green for in, red for fault — with service box labels.

**Per-Point Highlight Clips.** Automatic video clip extraction for every detected point with pre and post padding.

**Session Persistence.** Coach feedback, flags, and preferences saved across matches. Each new match run loads previous session data.

### 1.8 Demo Workflow

```
Coach uploads YouTube URL or MP4
    │
FastAPI creates job → runs inference pipeline locally
Frontend polls /status every 5 seconds
    │
Step 01: ResNet50 court keypoints → 14 keypoints per frame
Step 02: YOLOv5 ball detection → ball position per frame
Step 03: YOLOv8n + court boundary filter → 2 players per frame
Step 04: Merge into FrameRecord per frame
Step 05: Homography → court coordinates
         Physics-based bounce + hit detection
         Point segmentation state machine
         In/out classification via court geometry
         Serve placement + fault type
         Coaching cards + match insights
Step 06: Per-point clip extraction
    │
CHECKPOINT
Coach reviews detected points on overlay video
Confirms, flags, or notes corrections on tennis calls
Feedback saved to SQLite instantly
    │
Results page loads
Side-by-side sync player (raw || overlay)
Point timeline with serve zones and end reasons
Serve placement chart
Error heatmap
Player movement heatmaps
Per-point coaching cards
Match insights
```

### 1.9 Impact

TennisIQ compresses days of manual footage review into minutes. It surfaces serve placement patterns, error clustering, and rally tendencies that would otherwise go unnoticed. It delivers Hawkeye-grade analytics — grounded in real court geometry — to college programs that could never afford Hawkeye hardware.

Over a season it returns 30–60 hours of coaching staff time and materially improves the quality and speed of strategic decisions. The system improves across matches as coach feedback accumulates in session storage.

### 1.10 Value Proposition

- **For the Head Coach.** Serve placement charts, error heatmaps, and per-point coaching cards generated automatically after every match. Strategic decisions backed by data rather than memory.
- **For Players.** Objective feedback on serve placement, rally patterns, and court coverage from their own match footage — not generic advice.
- **For the Program.** Hawkeye-grade analytics without Hawkeye hardware costs. Levels the playing field against better-resourced programs.
- **For Technical Evaluators.** Purpose-built tennis CV models, physics-based event detection, real court geometry classification. Not a YOLO fine-tuning wrapper — a purpose-designed tennis analytics engine.

### 1.11 MVP Scope

**In Scope**
- College tennis footage (YouTube URL or MP4)
- YOLOv5 ball detection (pretrained, inference only)
- ResNet50 court keypoint regression + homography (pretrained, inference only)
- YOLOv8n player detection with court boundary filtering
- Physics-based bounce and hit detection with heuristic scoring
- Point segmentation via state machine
- In/out classification using real court polygon geometry
- Serve placement by service box and fault type
- Coach checkpoint review of tennis calls (not bounding boxes)
- Side-by-side sync video player as demo centerpiece
- Per-point coaching cards
- Serve placement chart (top-down court diagram)
- Error heatmap (out bounce distribution in court space)
- Player movement heatmaps
- Per-point highlight clips
- Match-level strategic insights
- Session persistence across matches

**Out of Scope**
- Model training or fine-tuning on uploaded footage
- Ball speed estimation in real units (post-MVP)
- Player identity tracking across matches
- Doubles formation analysis
- Real-time inference during live matches
- Mobile application
- Multi-user authentication

---

## Section 2: Functional Requirements

*Pattern: (Actor) + (shall) + (action) + (condition/purpose).*

### Ingestion

| ID | Requirement |
|---|---|
| FR-01 | The system shall accept a YouTube URL or MP4 file upload from the coach via the web interface as the primary footage input. |
| FR-02 | The system shall attempt YouTube download via yt-dlp once and immediately surface the MP4 upload prompt on any failure without silent retries. |
| FR-03 | The system shall validate uploaded footage before processing begins and notify the coach immediately if the input is unsupported, corrupted, or unreadable. |
| FR-04 | The system shall read native FPS from OpenCV at pipeline start and use it for all kinematic calculations — speed, acceleration, and timestamp derivation — without re-encoding or normalizing the video. |

### Video Segmentation and Execution

| ID | Requirement |
|---|---|
| FR-04a | Given a video up to 6 minutes, the system shall automatically partition it into 10-second segments, register all segments with the backend, and process each segment as an independent Modal GPU invocation. |
| FR-04b | The system shall deliver the first segment's results to the frontend as soon as it completes, without waiting for remaining segments to finish. |
| FR-04c | The system shall persist per-segment completion status via the backend /status/update endpoint after each segment finishes, so that if a Modal function is interrupted, only incomplete segments need to be re-run. |
| FR-04d | The system shall merge results from all completed segments into a single unified output — frames.jsonl, events.json, points.json, overlay video, and clips — with correct cross-segment frame indexing and timestamps. |

### Court Detection and Homography

| ID | Requirement |
|---|---|
| FR-05 | The system shall load the ResNet50 court keypoint regression model from the pretrained checkpoint (keypoints_model.pth) at pipeline start and run keypoint inference on every frame of the uploaded video. |
| FR-06 | The system shall detect 14 court line intersection keypoints per frame outputting pixel coordinates for each. |
| FR-07 | The system shall compute a homography matrix from the detected keypoints every frame using the court reference geometry. |
| FR-08 | The system shall apply temporal stabilization to the homography — carrying the last reliable homography forward for up to 5 frames during brief occlusion windows — to prevent jitter in court-space projections. |
| FR-09 | The system shall compute a homography confidence score per frame based on reprojection error and temporal stability, and flag frames below the confidence threshold as unreliable for in/out classification. |

### Ball Tracking

| ID | Requirement |
|---|---|
| FR-10 | The system shall load the YOLOv5 tennis ball detection model from the pretrained checkpoint (models_best.pt) at pipeline start and run ball detection inference on every frame. |
| FR-11 | The system shall postprocess YOLOv5 bounding box outputs to extract ball center coordinates per frame, applying confidence thresholding and single-best-detection selection. |
| FR-12 | The system shall remove outlier detections where inter-frame ball distance exceeds a maximum threshold. |
| FR-13 | The system shall fill gaps in the ball track via linear interpolation within continuous sub-tracks, preserving None values across large gaps where interpolation is unreliable. |
| FR-14 | The system shall project ball pixel coordinates to court-space coordinates using the homography matrix every frame. |
| FR-15 | The system shall compute ball speed and acceleration in court-space coordinates per frame using finite differences. |

### Player Detection

| ID | Requirement |
|---|---|
| FR-16 | The system shall run YOLOv8n with ByteTrack on every frame to detect and track person bounding boxes. |
| FR-17 | The system shall filter detected persons using court boundary projection — any detection whose foot position projects outside the court doubles boundary plus a buffer margin shall be discarded as non-player noise. |
| FR-18 | The system shall select Player A as the largest valid detection with the lowest foot y-coordinate (near side) and Player B as the largest remaining valid detection (far side). |
| FR-19 | The system shall fall back to HOG-based person detection if YOLOv8n is unavailable, applying the same court boundary filter. |
| FR-20 | The system shall project player foot positions to court-space coordinates using the homography matrix every frame. |

### Event Detection

| ID | Requirement |
|---|---|
| FR-21 | The system shall detect bounce candidates by identifying frames where ball vertical velocity reverses direction combined with a speed drop relative to adjacent frames. |
| FR-22 | The system shall score bounce candidates using heuristic kinematic features — speed drop ratio, deceleration magnitude, vertical reversal strength — normalized to a 0–1 confidence score, and apply temporal non-maximum suppression to remove duplicate detections. |
| FR-23 | The system shall detect hit events by identifying frames where ball direction changes by 50 or more degrees with sufficient speed, applying non-maximum suppression by score. |
| FR-24 | The system shall classify every bounce event as in, out, or line using real court polygon geometry in court-space coordinates with a configurable line margin. |

### Point Segmentation

| ID | Requirement |
|---|---|
| FR-25 | The system shall segment continuous match footage into discrete tennis points using a state machine with states IDLE, SERVE SETUP, SERVE FLIGHT, and RALLY. |
| FR-26 | The system shall assign an end reason to every detected point — OUT, DOUBLE BOUNCE, NET, or BALL LOST — based on bounce classification and state machine transitions. |
| FR-27 | The system shall record start frame, end frame, serve frame, first bounce court coordinates, rally hit count, bounce frames, and confidence score for every detected point. |
| FR-28 | The system shall classify the serve zone for every point by mapping the first bounce court coordinate to the appropriate service box — top-left, top-right, bottom-left, or bottom-right. |
| FR-29 | The system shall classify serve fault type as wide, long, or net when the first bounce lands outside the valid service box. |

### Coach Checkpoint Review

| ID | Requirement |
|---|---|
| FR-30 | The system shall pause after inference and output generation and present the coach with the detected point list for review before finalizing the analytics package. |
| FR-31 | The coach shall be able to confirm, flag, or add a plain-text note to each detected point via the checkpoint review interface, reviewing tennis calls rather than bounding box labels. |
| FR-32 | The system shall display the overlay video alongside each detected point at checkpoint so the coach can visually verify the detection against actual footage. |
| FR-33 | The system shall write all coach checkpoint feedback to SQLite instantly upon submission. |
| FR-34 | The system shall apply coach feedback to adjust confidence thresholds and flagged point exclusions in the final analytics output. |

### Visual Output

| ID | Requirement |
|---|---|
| FR-35 | The system shall render a full overlay video with ball trajectory traced in yellow, court keypoints marked, Player A box in blue, Player B box in orange, and ball colored green for in and red for out. |
| FR-36 | The system shall generate a serve placement chart as a top-down court diagram with every serve plotted — green dot for in, red dot for fault — with service box regions labeled. |
| FR-37 | The system shall generate an error heatmap as a 2D histogram of all out bounce positions in court-space coordinates. |
| FR-38 | The system shall generate player movement heatmaps for Player A and Player B separately showing court coverage distribution in court-space coordinates. |
| FR-39 | The system shall extract a video clip for every detected point with one second of pre-padding and 0.2 seconds of post-padding. |
| FR-40 | The system shall generate a per-point coaching card for every detected point containing a plain-English explanation of why the point ended and what to try differently, grounded in the actual detection data for that point. |

### Dashboard and Delivery

| ID | Requirement |
|---|---|
| FR-41 | The coach shall land on the results page with the side-by-side sync video player as the primary visual — raw footage left, overlay right, synced on a single scrub bar. |
| FR-42 | The system shall implement side-by-side sync via two HTML5 video elements driven by a single scrub bar with JavaScript timeupdate synchronization. |
| FR-43 | Clicking any point in the timeline shall jump both videos simultaneously to the start frame of that point. |
| FR-44 | The frontend shall poll /status every 5 seconds during the pipeline run and display a live stage tracker with current step name and plain-English status description. |
| FR-45 | The coach shall be able to view the serve placement chart, error heatmap, player movement heatmaps, point timeline, per-point clips, and coaching cards from the same results page without navigating away. |
| FR-46 | The coach shall be able to download the overlay video, points JSON, insights JSON, and all highlight clips from the results page. |

### Session Persistence

| ID | Requirement |
|---|---|
| FR-47 | The system shall save a complete session record after every pipeline run including footage metadata, coach feedback, detection results, and match insights. |
| FR-48 | The system shall load session data at the start of subsequent pipeline runs for the same coach, applying saved preferences and flagged corrections as starting state. |

---

## Section 3: Non-Functional Requirements

### Performance

| ID | Requirement |
|---|---|
| NFR-P01 | Given a video up to 6 minutes, the system shall automatically partition it into 10-second segments and deliver the first segment's results in ≤ 30 seconds on a single Modal T4, while completing the remaining segments asynchronously. |
| NFR-P02 | Homography computation shall add no more than 5 milliseconds per frame overhead above inference time. |
| NFR-P03 | The results page shall load annotated video, charts, and clips within 5 seconds of pipeline completion. |
| NFR-P04 | The side-by-side sync player shall maintain frame-accurate synchronization between raw and overlay video at all scrub positions. |
| NFR-P05 | Per-point clip extraction shall complete within 60 seconds for a full match regardless of point count. |

### Reliability

| ID | Requirement |
|---|---|
| NFR-R01 | If YouTube download fails, the system shall notify the coach immediately and prompt MP4 upload without further retry attempts. |
| NFR-R02 | If ball tracking loses the ball for more than 4 consecutive frames, the system shall apply gap-fill interpolation within sub-tracks and preserve None values across larger gaps rather than hallucinating positions. |
| NFR-R03 | If homography confidence falls below threshold for more than 5 consecutive frames, the system shall flag any points whose frame range overlaps that drop as low-confidence, penalizing their confidence score proportionally rather than producing unreliable in/out calls. |
| NFR-R04 | If the YOLOv8n model fails to load or crashes, the system shall fall back to OpenCV HOG-based person detection automatically so the pipeline continues without blocking. If HOG also produces zero detections, the frame shall carry forward the last valid player positions. |
| NFR-R05 | If court boundary filtering eliminates all person detections in a frame, the system shall carry forward the last valid Player A and Player B positions rather than producing null entries that corrupt heatmap and event data. |
| NFR-R06 | The Modal pipeline shall persist per-segment completion status via the backend /status/update endpoint after each 10-second segment finishes, so that if the Modal function is interrupted, only the incomplete segment needs to be re-run rather than restarting the full video. |

### Usability

| ID | Requirement |
|---|---|
| NFR-U01 | A coach with no machine learning background shall be able to upload footage, complete checkpoint review, and access final outputs without any technical instructions beyond the UI. |
| NFR-U02 | The checkpoint review interface shall present detected points as tennis events — start time, end reason, serve zone, rally length — not as bounding box images requiring ML judgment. |
| NFR-U03 | All coaching cards shall be written in plain English with no ML or CV terminology. |
| NFR-U04 | The serve placement chart shall be readable as a standard tennis court diagram without any explanation. |
| NFR-U05 | All error messages shall describe what went wrong and what action to take next. |
| NFR-U06 | The pipeline progress tracker shall display current step name and plain-English description at all times. |

### Security

| ID | Requirement |
|---|---|
| NFR-S01 | Uploaded footage shall not be stored beyond the active pipeline session without explicit coach consent. |
| NFR-S02 | All data transmitted between frontend and backend shall be encrypted in transit via HTTPS. |

---

## Section 4: System Design and Architecture

### 4.1 High-Level Structure

```
+--------------------------------------------------+
|          NEXT.JS FRONTEND (localhost:3000)        |
|                                                  |
|  +--------------------------------------------+ |
|  |    SIDE-BY-SIDE SYNC VIDEO PLAYER          | |
|  |  Raw Footage  ||  TennisIQ Overlay         | |
|  |  [══════●════]||[══════●════]              | |
|  |      Single scrub bar — synced             | |
|  +--------------------------------------------+ |
|                                                  |
|  Upload → Progress → Checkpoint → Results        |
|  Serve Chart · Error Heatmap · Coaching Cards    |
+---------------------------+----------------------+
                            | REST /status poll 5s
+---------------------------+----------------------+
|        FASTAPI BACKEND (localhost:8000)          |
|                                                  |
|  /ingest     Accept video, create job in SQLite  |
|  /status     Return current step + progress      |
|  /checkpoint Receive coach feedback → SQLite     |
|  /results    Return artifact paths               |
|                                                  |
|  Calls tennisiq/pipeline/run_all.py directly     |
+---------------------------+----------------------+
                            |
+---------------------------+----------------------+
|     TENNISIQ INFERENCE PIPELINE                  |
|                                                  |
|  Step 01: ResNet50 court keypoint inference       |
|           14 keypoints per frame                 |
|           Homography per frame                   |
|                                                  |
|  Step 02: YOLOv5 ball detection inference        |
|           Ball (x,y) per frame                   |
|           Gap fill + outlier removal             |
|                                                  |
|  Step 03: YOLOv8n + court boundary filter        |
|           Player A + Player B per frame          |
|                                                  |
|  Step 04: FrameRecord merge                      |
|           All detections joined per frame        |
|                                                  |
|  Step 05: Analytics engine                       |
|           Court-space projection                 |
|           Bounce + hit detection                 |
|           Point segmentation state machine       |
|           In/out classification                  |
|           Serve placement + fault type           |
|           Coaching cards + insights              |
|                                                  |
|  Step 06: Clip extraction                        |
|           Per-point MP4 clips                    |
|                                                  |
|  Outputs written to outputs/runs/<job_id>/       |
+--------------------------------------------------+
              |
+-------------+----------+
|  PRETRAINED CHECKPOINTS |
|  checkpoints/models_best.pt       YOLOv5 ball     |
|  checkpoints/keypoints_model.pth ResNet50 court  |
+------------------------+
```

### 4.2 Data Flow Per Frame

```
Raw video frame
    │
    ├─ ResNet50 keypoints ─→ 14 keypoints (pixel coordinates)
    │                            │
    │                       Homography matrix
    │                            │
    ├─ YOLOv5 ball det. ─→ ball_xy (pixel) → ball_court_xy (real court)
    │                                         ball_speed, ball_accel
    │                                         ball_inout (in/out/line)
    │
    └─ YOLOv8n ─────────→ person boxes
                               │
                         court boundary filter
                         (discard ball boys, line judges, crowd)
                               │
                         playerA_bbox, playerB_bbox
                         playerA_court_xy, playerB_court_xy
    │
    ▼
FrameRecord
  frame_idx, ts_sec
  court_keypoints, homography_ok, homography_confidence
  ball_xy, ball_court_xy, ball_visible
  ball_speed, ball_accel, ball_inout
  playerA_bbox, playerA_court_xy
  playerB_bbox, playerB_court_xy
  event_candidates, event_scores
```

### 4.3 Analytics Engine Detail

```
All FrameRecords
    │
Bounce candidate detection
  (velocity reversal + speed drop per frame)
    │
Heuristic bounce scoring
  (kinematic features → normalized 0–1 confidence score)
    │
Hit event detection
  (direction change 50°+ + speed threshold)
    │
Point segmentation state machine
  IDLE → ball appears → SERVE SETUP
  SERVE SETUP → hit detected → SERVE FLIGHT
  SERVE FLIGHT → bounce detected → RALLY
  RALLY → out bounce → end reason OUT
  RALLY → double bounce same side → DOUBLE BOUNCE
  RALLY → bounce near net → NET
  RALLY → ball lost 12+ frames → BALL LOST
    │
Per-point serve zone classification
  first_bounce_court_xy → which service box → IN / FAULT TYPE
    │
Aggregation
  serve_in_pct
  fault type distribution (wide / long / net)
  error cluster side (deuce / ad)
  end reason distribution
  rally length distribution
    │
Coaching cards per point
  end_reason + serve_zone + rally_length + confidence
  → plain English why + try instead
    │
Match insights
  "Your serves miss mostly wide."
  "Your errors cluster on the deuce side."
```

### 4.4 Checkpoint Review Detail

```
Inference completes → results written to disk
    │
SQLite job status → awaiting_review
Frontend poll detects → shows checkpoint UI
    │
Coach sees:
  Point timeline list with timestamps + end reasons + serve zones
  Overlay video player for spot-checking any point
    │
Coach actions per point:
  ✓ Confirm — detection looks correct
  ✗ Flag — wrong call or missed point
  ✎ Note — plain text context ("double fault not a miss")
    │
Coach submits → POST /checkpoint
→ SQLite updated instantly
→ Flagged points excluded from analytics
→ Notes saved to session.json
→ Results page unlocks
```

### 4.5 Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Frontend | Next.js + Tailwind CSS | Fast build, handles sync player cleanly |
| Backend API | FastAPI (Python) | Wraps existing Python pipeline directly |
| Inference | Modal T4 GPU | Serverless GPU inference — no persistent infra |
| Job State | SQLite | Instant reads/writes, zero infra setup |
| Ball Tracking | YOLOv5 (models_best.pt, pretrained) | Fine-tuned for tennis ball, handles motion blur |
| Court Detection | ResNet50 keypoint regression (keypoints_model.pth, pretrained) | 14 keypoints → homography → real court coordinates |
| Player Detection | YOLOv8n + court boundary filter | General detector constrained by court geometry |
| Event Detection | Physics-based + heuristic scoring | Bounce/hit from kinematics, not pixel patterns |
| Video Ingestion | yt-dlp + OpenCV | YouTube + MP4, fail fast on error |
| Overlay Rendering | OpenCV + FFmpeg | Pre-renders annotated video for sync player |
| Charts | JSON data + frontend SVG | Serve placement, error heatmap, player heatmaps rendered in browser |
| Progress Updates | REST polling /status (5s) | Simple, reliable, no WebSocket complexity |

### 4.6 Project Structure

```
tennisiq/
  modal_court.py                   # Modal GPU pipeline orchestrator
  analytics/
    __init__.py                    # Exposes detect_events, segment_points
    events.py                      # Bounce/hit detection, scoring, NMS
    points.py                      # Point segmentation state machine
  cv/
    ball/
      inference.py                 # BallTrackerNet (legacy)
      inference_yolo5.py           # YOLOv5 ball detection (active)
      model.py                     # BallTrackerNet architecture (legacy)
    court/
      inference.py                 # Court keypoint (legacy)
      inference_resnet.py          # ResNet50 keypoint regression (active)
      model.py                     # CourtKeypointNet architecture (legacy)
      model_resnet.py              # ResNet50 architecture (active)
    players/
      inference.py                 # YOLOv8n + ByteTrack, HOG fallback
  geometry/
    court_reference.py             # Standard court dimensions, polygons
    homography.py                  # Homography computation + stabilization
  io/
    video.py                       # OpenCV video read, FPS extraction
    output.py                      # JSON output writers (events, points, heatmaps, cards)
    visualize.py                   # Overlay rendering, point clip extraction

pipeline/
  stage_00_load.py … stage_08_output.py  # Legacy training pipeline stages

backend/
  main.py                          # FastAPI: /ingest /status /checkpoint /results /segments
  db.py                            # SQLite job state, segments, point feedback
  modal_runner.py                  # Launches Modal pipeline from backend

frontend/
  app/
    page.tsx                       # Ingest page (YouTube URL / MP4 upload)
    layout.tsx                     # Root layout
    results/[jobId]/page.tsx       # Results dashboard
  components/
    ProgressTracker.tsx            # Live step tracker
    CheckpointReview.tsx           # Point confirm/flag/note UI
    SideBySidePlayer.tsx           # Synced dual video player
    ServePlacementChart.tsx        # Top-down court serve placement (SVG)
    HeatmapViewer.tsx              # Error + player heatmaps (SVG)
    PointTimeline.tsx              # Scrollable point list with seek
    CoachingCards.tsx              # Per-point why + try instead
    HighlightClips.tsx             # Per-point video clips
    DownloadPanel.tsx              # Artifact download links
  lib/
    api.ts                         # API client functions
    types.ts                       # TypeScript interfaces

checkpoints/
  keypoints_model.pth              # ResNet50 court keypoints
  models_best.pt                    # YOLOv5 tennis ball

outputs/<run_id>/
  overlay.mp4                      # Annotated video for sync player
  frames.jsonl                     # Per-frame detection data
  events.json                      # Detected bounces and hits
  points.json                      # Point segmentation + serve zones
  stats.json                       # Aggregate statistics
  run.json                         # FPS, frame count, run metadata
  visuals/
    ball_heatmap.json              # Ball position heatmap data
    player_coverage.json           # Player A/B court coverage data
    speed_histogram.json           # Ball speed distribution
  timeseries/
    ball_court.json                # Ball court-space trajectory
    player_a_court.json            # Player A court-space trajectory
    player_b_court.json            # Player B court-space trajectory
```

### 4.7 Fallback Strategy

| Intended Behavior | Fallback |
|---|---|
| YouTube download via yt-dlp | Fail immediately, prompt MP4 upload |
| Homography computed every frame | Carry last reliable homography up to 5 frames |
| Ball tracked every frame | Gap-fill interpolation within sub-tracks |
| YOLOv8n player detection | Fall back to HOG detector automatically |
| Court boundary filter finds 2 players | Carry last valid player positions forward |
| Heuristic bounce scoring consistent | Tunable thresholds for speed drop, deceleration, reversal |
| High homography confidence | Flag low-confidence frames, exclude from in/out calls |
| Pipeline completes without interruption | SQLite step checkpoint — resume from last step |
