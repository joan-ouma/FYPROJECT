# Computer Vision (Objective 1)

Owner: Ouma Joan Akello

**Goal:** recognise the 5 products on the shelf with mAP@0.5 of at least 90% using the
robot's own camera, run at 5 or more frames per second on the Raspberry Pi 5, and give
each item's position to the arm within ±1.5 cm.

**How it works:** the camera photo goes to a YOLOv8n model, which finds the products.
A product only counts once it is seen at confidence 0.6 or more in 3 frames in a row.
An ArUco marker on the shelf slot gives the exact 3D position of that slot. The item's
position is the marker position plus a measured offset, nudged left or right by where the
model saw the item. Finally it is converted into the arm's coordinates for Joy's code.

## Folder layout

```
vision/
  products.txt              the 5 product names: egg, tea_leaves, matchbox, biscuits, toothpaste
  requirements.txt          Python packages
  config/
    vision_config.yaml      all settings (camera, model, confidence, marker size, ...)
    shelf_slots.yaml        which marker is on which slot + where the item sits from it
    camera_to_arm.yaml      where the camera is compared with the arm base
  shopvision/               the code library (the rest of the robot imports this)
    service.py              VisionService.find_item("egg")  <- the main function
    detector.py confirm.py aruco.py locator.py transforms.py camera.py config.py metrics.py
  capture/
    capture_images.py       take photos (dataset or calibration)
    dataset_status.py       how many photos you have
  calibration/
    make_checkerboard.py    printable A4 checkerboard
    calibrate_camera.py     works out the camera's lens settings
  markers/
    make_markers.py         printable ArUco markers for the shelf slots
  training/
    train_on_colab.ipynb    open in Google Colab: check, train, score, export - all in one
    check_dataset.py        checks the Roboflow export (names, order, counts)
    train.py                trains YOLOv8n, scores it, writes a results table
    evaluate.py             scores any model on the test photos
    export_model.py         converts best.pt to NCNN for the Pi
  tools/
    live_detect.py          live view with boxes, markers, FPS
    benchmark_fps.py        measures frames per second (target 5)
    test_aruco_accuracy.py  measures marker position error (target 15 mm)
    measure_camera_to_arm.py  works out camera_to_arm.yaml
    find_item.py            runs the full step once (or 20 times for a report test)
  ros2/
    vision_node.py          ROS 2 node for mission control (not tested on the robot yet)
  tests/                    automatic tests: python3 -m pytest tests
  models/                   trained models go here
  dataset/                  photos (NOT uploaded to GitHub, too big)
```

## Plan (November - December 2026)

| Week | Dates | Task | Done when |
|---|---|---|---|
| 1 | Nov 5-11 | Buy the 5 products; agree camera position with Yvonne and Joy; build a 2-level cardboard shelf | Done: products bought |
| 2 | Nov 12-18 | Pi camera working; print checkerboard; calibrate | `camera_calibration.yaml` saved, error under 0.5 px |
| 3-4 | Nov 19 - Dec 2 | Take about 150 photos with several products in each; label in Roboflow | Each product appears 80+ times |
| 5 | Dec 3-9 | Train on Colab; add photos for weak products | mAP@0.5 of 90% or more |
| 6 | Dec 10-16 | Markers on shelf slots; accuracy test; camera-to-arm | Error within 15 mm |
| 7 | Dec 17-23 | NCNN model on the Pi; FPS test; `find_item` repeat test | 5+ FPS, found 18/20 times or better |
| 8 | Dec 24-31 | Spare week; write up results | Results tables ready |

## 1. Set up the Raspberry Pi (once)

Use Raspberry Pi OS Bookworm (64-bit). Picamera2 comes with it.

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-venv
python3 -m venv --system-site-packages ~/vision-env   # system packages = Picamera2
source ~/vision-env/bin/activate
cd FYPROJECT/vision
pip install -r requirements.txt
pip install ultralytics ncnn        # only needed on the robot, to run the model
rpicam-hello -t 2000                # quick check that the camera works
python3 -m pytest tests             # all tests should pass
```

On a laptop, run `pip install -r requirements.txt` (and `pip install ultralytics` for the model).

## 2. Calibrate the camera

1. `python3 calibration/make_checkerboard.py` makes `checkerboard_9x6_25mm.pdf`. Print it at
   **100% / actual size** and measure one square with a ruler. Glue it flat on stiff cardboard.
2. Mount the camera exactly where it will be on the robot. For a Camera Module 3, lock the
   focus and use the SAME value everywhere (also in `config/vision_config.yaml` -> `lens_position`):
   ```bash
   python3 capture/capture_images.py --label calib --lens-position 2.0
   ```
   Take 20-30 photos (SPACE). Move the board near, far, tilted, and into each corner.
3. `python3 calibration/calibrate_camera.py --square-mm 25 --check`
   An error under 0.5 px is very good. Over 1.0 px: delete the "worst photos" it lists and run again.

## 3. Take the product photos (about 150, several products in each)

```bash
python3 capture/capture_images.py --label mixed_morning                       # SPACE saves, Q quits
python3 capture/capture_images.py --label mixed_night --every 1.5 --count 40   # one photo every 1.5 s
python3 capture/dataset_status.py
```

Move the products around between photos. Use the same camera position, size and focus as for calibration.

**Checklist - mix all of these:**

- [ ] Light: morning, afternoon, evening, room light on and off, a torch from the side
- [ ] Position: left, centre, right; top shelf and bottom shelf
- [ ] Angle: products turned slightly left and right, front and side
- [ ] Neighbours: products next to each other in different orders
- [ ] Partly hidden: one item covering up to a third of another
- [ ] Distance: the real distance from the robot, plus a little nearer and further
- [ ] A few photos of an empty shelf (no labels)
- [ ] Egg: many angles, lying and standing, next to white and brown things
- [ ] Matchbox and toothpaste: every side (front, back, ends)

**Extra photos from Kaggle or Roboflow Universe** can go in TRAINING only. Keep the TEST set
to your own camera photos, rename their classes to match `products.txt`, and note the licence.

## 4. Label and train

1. Upload the photos to Roboflow. Create the 5 classes **in the same order as `products.txt`**.
   Draw a box around every product in every photo.
2. Generate a version: split about 70% train / 20% valid / 10% test. Add augmentation
   (brightness ±25%, blur up to 1 px, rotation ±10°) to get about 3x the training photos.
3. Open `training/train_on_colab.ipynb` in Google Colab (File -> Open notebook -> GitHub tab,
   or upload it), choose a T4 GPU, and run the cells from top to bottom. It checks the dataset,
   trains, scores the model on the test photos, converts it to NCNN and saves it to your Drive.
4. Read `training/results/<run>_summary.md`. Any product below 0.90 needs more photos:
   take them in the light and angles it got wrong, then train again (write what changed in `--notes`).

## 5. Markers, accuracy and the arm

1. `python3 markers/make_markers.py` makes `markers/aruco_markers.pdf` (IDs 0-4, 40 mm). Print at
   100%, measure the black square, and put that number in `vision_config.yaml` -> `marker_size_mm`.
   **This number matters:** a 10 mm mistake here gives about 9 cm of position error.
2. Stick one marker flat on the front edge of the shelf under each slot. In `config/shelf_slots.yaml`,
   set which product each marker ID is for, and measure `item_offset_mm` (from the marker centre
   to the item centre: right, up, back).
3. Accuracy test (target 15 mm): `python3 tools/test_aruco_accuracy.py --marker-id 0`.
   Put the marker at measured distances (250, 350, 450 mm, straight and turned), type each real distance.
4. With Joy, agree the arm frame (origin at the arm base, x forward, y left, z up). Then lay a marker
   flat at a measured spot and run `python3 tools/measure_camera_to_arm.py --marker-at 200 0 0`.

## 6. Run it on the robot

```bash
python3 tools/live_detect.py --target egg          # see boxes, markers, FPS, confirmation
python3 tools/benchmark_fps.py --with-markers      # FPS test (target 5)
python3 tools/find_item.py egg                     # full step once, prints the arm position
python3 tools/find_item.py egg --repeat 20         # repeat test for the report
```

If FPS is under 5, export at a smaller size (`training/export_model.py --imgsz 416`), set
`image_size: 416` in the config and check the score again with `training/evaluate.py --imgsz 416`.

**For the rest of the robot** (mission control / behaviour tree):

```python
from shopvision.service import VisionService
vision = VisionService()
r = vision.find_item("egg")      # {"status": "found", "found": True, "position_arm_m": [x, y, z], ...}
```

or over ROS 2: run `python3 ros2/vision_node.py`, publish the product name on `/vision/find_item`,
and read the JSON answer on `/vision/item_result`.

## What is tested so far

`python3 -m pytest tests` runs 20 tests on made-up camera pictures: marker position (errors of
0.1-2 mm), the item locator, the arm-frame conversion, the camera-to-arm tool, the confirmation
rule, the full `find_item` with a stand-in model, the dataset checker and the printed markers.
**Not yet tested on real hardware:** the Pi camera, the YOLO model (training, export, live
detection) and the ROS 2 node. Report any errors from those and they will be fixed.

## Rules we agreed (lessons from RECAP)

- Every photo is taken with **the camera that goes on the robot, from where it is mounted**.
  RECAP's model scored 98% on one camera and failed on another.
- One YOLO version everywhere: **YOLOv8n**.
- If the camera position or focus changes, calibrate again (and take new photos if it changed a lot).
