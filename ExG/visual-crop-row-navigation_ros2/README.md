# Visual Crop Row Navigation

<div align="center">
	<img src=".readme/vs_poster.png" alt="visual_servoing_husky" height="180" title="visual_servoing_husky"/>
</div>


**Update**

A python based implementation for Multi-crop-row navigation can be found here [visual-multi-crop-row-navigation](https://github.com/Agricultural-Robotics-Bonn/visual-multi-crop-row-navigation)

<div align="center">
	
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/z2Cb2FFZ2aU/0.jpg)](https://www.youtube.com/watch?v=z2Cb2FFZ2aU)
	
</div>

This is a visual-servoing based robot navigation framework tailored for navigating in row-crop fields.
It uses the images from two on-board cameras and exploits the regular crop-row structure present in the fields for navigation, without performing explicit localization or mapping. It allows the robot to follow the crop-rows accurately and handles the switch to the next row seamlessly within the same framework.

This implementation uses C++ and ROS and has been tested in different environments both in simulation and in real world and on diverse robotic platforms.

This work has been developed @ [IPB](http://www.ipb.uni-bonn.de/), University of Bonn.

Check out the [video1](https://www.youtube.com/watch?v=uO6cgBqKBas), [video2](https://youtu.be/KkCVQAhzS4g) of our robot following this approach to navigate on a test row-crop field.

<div align="center">
	<a href="http://www.youtube.com/watch?feature=player_embedded&v=0qg6n4sshHk
		" target="_blank"><img src=".readme/husky_test.gif" alt="husky_navigation" height="250" title="husky_navigation" border="5"/><img src=".readme/husky_test_nav.gif" alt="husky_navigation" height="250" title="husky_navigation" border="5"/></a>
	<!-- <a href="http://www.youtube.com/watch?feature=player_embedded&v=0qg6n4sshHk
		" target="_blank"><img src="http://img.youtube.com/vi/0qg6n4sshHk/0.jpg"
		alt="Watch video" height="250" border="10" /></a> -->
</div>


## Features

- No maps or localization required.
- Running on embedded controllers with limit processing power (Odroid, Raspberry Pi).
- Simulation environment in Gazebo.
- Robot and cameras agnostic.

## Robotic setup

This navigation framework is designed for mobile robots equipped with two cameras mounted respectively looking to the front and to the back of the robot as illustrated in the picture below.

 <div align="center">
	<img src=".readme/vs_graph.png" alt="agribot_3d" height="250" title="agribot_3d"/>
    <img src=".readme/vs_em.png" alt="camera_img" height="250" title="camera_img"/>
</div>

A complete Gazebo simulation package is provided in [agribot_robot](https://github.com/PRBonn/agribot) repository including simulated row-crop fields and robot for testing the navigation framework.

<div align="center">
	<img src=".readme/motivation.png" alt="husky_navigation" height="280" title="husky_navigation"/>
    <img src=".readme/motivation_old.png" alt="gazebo_navigation" height="280"title="gazebo_navigation"/>
</div>

## Dependencies

- c++17
- ROS2 (tested with Humble)
- opencv >= 3
- Eigen >= 3.3

## How to build and run

1. Clone the package into your *ros2_ws*
```bash
cd ~/ros2_ws/src
git clone https://github.com/PRBonn/visual_crop_row_navigation.git visual-crop-row-navigation_ros2
```
2. Build the package
```bash
cd ~/ros2_ws
colcon build --packages-select visual_crop_row_navigation_ros2
source install/setup.bash
```
3. Install the USB camera driver (ROS2 version of usb_cam)
```bash
sudo apt install ros-humble-usb-cam
```
4. Run visual servoing navigation
```bash
ros2 launch visual_crop_row_navigation_ros2 visualservoing.launch
```

Successfully tested using:
- Ubuntu 22.04
- ROS2 Humble

## Test data

Download the bagfile used for our experiments [here]().

## Simulation 

Simultion and robot packages can be found in [Agribot repo](https://github.com/PRBonn/agribot)

---

## Citation 
if you use this project in your recent works please refernce to it by:

```bash

@article{ahmadi2021towards,
  title={Towards Autonomous Crop-Agnostic Visual Navigation in Arable Fields},
  author={Ahmadi, Alireza and Halstead, Michael and McCool, Chris},
  journal={arXiv preprint arXiv:2109.11936},
  year={2021}
}

@inproceedings{ahmadi2020visual,
  title={Visual servoing-based navigation for monitoring row-crop fields},
  author={Ahmadi, Alireza and Nardi, Lorenzo and Chebrolu, Nived and Stachniss, Cyrill},
  booktitle={2020 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={4920--4926},
  year={2020},
  organization={IEEE}
}

```

## Acknowledgments
This work has been supported by the German Research Foundation under Germany’s Excellence Strategy, EXC-2070 - 390732324 ([PhenoRob](http://www.phenorob.de/)) and [Bonn AgRobotics Group](http://agrobotics.uni-bonn.de/)

---

## Offline Python Debug Pipeline & Robustness Improvements (2026-08)

This fork adds a **stage-by-stage Python debug visualizer** that replicates the C++ visual-servoing core `src/agribot_vs.cpp:153` `CropRowFeatures` / `203` `getContureCenters` / `673` `is_in_neigbourhood` / `247` `FitLineOnContures` and `src/agribot_vs_nodehandler.cpp:49` `CropRow_Tracking()` outside ROS. It is used to locate failures per stage, mirroring `ClusterAlg/run_carolif.py` `4×4` debug composites.

**Debug visualizer:** `results/run_vcrn_debug.py` + `results/*_debug.png` (`5×4` `28×24` `130dpi`) and `results/run_exg_video.py` for video.

Pipeline per image (640×480 `params/agribot_vs_run.yaml:45`):

```
01 Original (full res) -> 02 Resized 640×480 (pipeline input) -> 03 Hue Mask H40-80 -> 04 Sat Mask S50-255 -> 05 Val Mask V100-150 -> 06 Combined H&S&V (173) -> 07 Contours findContours N -> 08 All Centers approxPolyDP+minEnclosingCircle -> 09 Column Profile (bottom 45% ROI, Gaussian sigma5, find_peaks distance~28 prominence0.12) -> 10 Window dynamic L×H @Xc,Yc (cyan) -> 11 Inside Window YELLOW nh | before IF (gray outside) -> 12 IsolationForest (YELLOW inliers RED outliers) -> 13 Raw fitLine BLUE infinite (fitLine:247) -> 14 Clipped AvgLine RED (is_in_image_point:285) -> 15 Final GREEN all + YEL + RED -> 16 Rejection / Notes -> 17 Summary
```

Original `results/README.txt` overlays (`green dots` centres `red` steering line `orange` window) are preserved as `15 Final`; the new composite exposes every decision point. `run_exg_video.py` reuses the same pipeline per video frame with fixed 640×480 writer and outputs `*_exg_nav.mp4` and `*_exg_nav.csv`.

### Improvements over upstream `origin/main`

**1. Window repositioned and thinned - closer to chassis:** upstream `ex_Xc320 ex_Yc240 nh_L120 nh_H250` centred mid-image `115-365` captures 2-3 converging rows (vanishing point) and dilutes `fitLine`. Tuned `ex_Xc320 ex_Yc380 nh_L80 nh_H180` `280-360 × 290-470` (`params:47`) pulls bottom edge to `y470` near robot chassis, thins `120->80` (`0.65× median inter-row gap`). Average `nh_points` `272->~110` (e.g. `photo_3` `649->165`, `bev` `378->102`), single-row isolation. `09 Column Profile` verifies.

**2. Isolation Forest inside window only:** `IsolationForest contamination0.15 n_estimators100` (`params:55`) on `nh_points` `x,y` inside window `n>=12`. Removes sparse weed/outlier `~15%` (e.g. `bev` `33/218`, `photo_3` `33/215` red) before `fitLine`. Skipped if `n<12`. Visualized `12` `YELLOW` inliers `RED` outliers, logged `IsolationForest inside window: removed X/Y`.

**3. Column-aware dynamic spawning - window locks onto a row:** fixed `Xc320` misses rows when robot off-centre (`photo_11` `0` points at `320` vs `5` at `162`) or sits in furrow scraping two rows. New `detect_column_aware_window()` `run_vcrn_debug.py:150` computes `profile[x]=sum(combined[y0:,x])` `y0=0.55*H` bottom `45%`, smooths, `find_peaks distance28`, `median_gap -> L_dyn=clip(median_gap*0.65,60,110)`, chooses `peak closest to image centre 320` weighted by prominence `score=|x-320|-30*prom`. `Xc=clip(chosen, L/2, W-L/2)`. `09` shows white profile, red peaks green chosen, cyan band. `photo_11` now `3 peaks gap266 chosen162 -> 80x180 @162,380` `5` points recovered `YES` vs `0` before; `bev` `12 peaks gap51 chosen304 -> 60x180 @304,380`. Fallback to static `Xc320` if no peaks.

**4. Gap-based multi-row filter inside window - evaluated and removed:** `KMeans K=2` on `x` with gap significance `gap>thresh` was tested for `photo_3` `2` dense rows `28px` `kept 73/140` vs `DBSCAN eps14`. It correctly split `2` rows when window was wide `120`, but with the tuned thin column-aware `60-80` window it consistently yielded `1` cluster for all `23` images (`gap 8 < thresh 24` `bev`), safety net not needed. Per user request the `KMeans` gap filter was removed and disabled `gap_enabled false:63`, `IsolationForest` remains the sole anomaly filter inside window. The `filter_gap_clusters()` code is retained dormant for future use. Timings now `resize/hsv/contours/centers/win/iso/fit` `gap 0`.

All filters **inside window only** (`gray` outside never scored), `Yc380` base retained.

### Params (`params/agribot_vs_run.yaml`)

```yaml
width: 640
height: 480
ex_Xc: 320  # static fallback, dynamic overrides Xc
ex_Yc: 380  # base
nh_L: 80    # fallback, dynamic L= median_gap*0.65
nh_H: 180
iso_enabled: true
iso_contamination: 0.15
iso_min_points: 12
colaware_enabled: true
colaware_y0_frac: 0.55
gap_enabled: false  # KMeans removed per request, stick to IsolationForest only
gap_eps: 14
gap_min_samples: 6
gap_min_points: 12
```

Upstream `filterContures` remains degenerate `center_min_off=0:38` noted in debug, window is effective filter.

### Results

`results/*_debug.png` (`23` images `Photos/bev*.png` `photo_*.png`) `5×4` `15+2` composites `09` column profile `12` IsolationForest, gap disabled. e.g. `bev_debug.png` `12 peaks gap51 Xc304 L60 102->86 IF->86` `photo_3` `7 peaks gap60 Xc326 L60 165->140 IF->140` `photo_11` `3 peaks gap266 Xc162 L80 5` recovered. Only `bev6` `0` (`H40-80` empty). Average `nh` `272->102` single-row purity.

`results/run_exg_video.py` on `Photos/test_video1.mp4` `700x370 480` frames and `test_video2.mp4` `960x540 558` frames with same pipeline `640x480` `~150ms` per frame `~30ms` after `IsolationForest` `~20ms` for `bev`, outputs `*_exg_nav.mp4` `640x480` overlay `v,w,err` and `*_exg_nav.csv`.

### Usage

```bash
python3 results/run_vcrn_debug.py --input ../Photos --output ./results --params params/agribot_vs_run.yaml
python3 results/run_exg_video.py --input ../../Photos/test_video1.mp4 --output ./video_output --show
# Images: --input Photos --output results --pattern "*.png"
# Video: --input test_video.mp4 --output video_output --show for live preview
# Tunable via yaml
```

C++ node unchanged; Python replicates `CropRow_Tracking()` for offline analysis. `src/agribot_vs.cpp` / `agribot_vs_nodehandler.cpp` / `agribot_types.h` preserved verbatim.
