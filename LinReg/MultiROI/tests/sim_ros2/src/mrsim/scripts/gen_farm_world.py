#!/usr/bin/env python3
"""Generate a clean multi-row maize Gazebo world from PRBonn agribot plant models.

Shapes:
  straight - rows run along +x, spaced `spacing` apart, plants spaced
             `plant_spacing` along each row plus noise. By default the rows
             follow a gentle lateral S-bend (furrow center
             y_c = curve_amp*sin(2*pi*(x-start_x)/curve_period)) so the rover
             must actively steer; pass --curve-amp 0 for straight rows.
  circle   - two concentric circular rows (radius --circle-r +/- spacing/2)
             around (--circle-cx, --circle-cy); the rover loops the circular
             furrow forever (lap counting in the nav node).

Plants are model://big_plant meshes (from the agribot repo models dir on
GAZEBO_MODEL_PATH), so the world file stays small and the sim stays fast.

Writes <out> world XML (default: ../worlds/farm_maize.world next to this
file) plus a sibling <stem>.spawn.json sidecar with the matching spawn pose
and lane params, which farm.launch.py uses as its argument defaults.
"""
import argparse, json, math, random, sys
from pathlib import Path

def fmt(v):  # compact float
    return ("%g" % v)

def plant_xml(m, name, x, y):
    return (
        f'      <include>\n'
        f'        <static>1</static>\n'
        f'        <uri>model://{m}</uri>\n'
        f'        <name>{name}</name>\n'
        f'        <pose>{fmt(x)} {fmt(y)} 0 0 0 {fmt(random.gauss(0.0, 0.06))}</pose>\n'
        f'      </include>\n'
    )

def main():
    ap = argparse.ArgumentParser()
    # Proven defaults (validated with the MultiROI closed loop in Gazebo): two
    # tall parallel rows flanking a single furrow, robot drives down x at y=0.
    ap.add_argument("--rows", type=int, default=2, help="number of parallel plant rows")
    ap.add_argument("--spacing", type=float, default=1.1, help="row spacing (m)")
    ap.add_argument("--length", type=float, default=18.0, help="row length along x (m)")
    ap.add_argument("--plant-spacing", type=float, default=0.12, help="plant spacing along row (m)")
    ap.add_argument("--noise", type=float, default=0.03, help="lateral/along noise (m)")
    ap.add_argument("--plant-scale", type=float, default=1.6,
                    help="horizontal (x/y) scale of the plant mesh (big_plant STL "
                         "is ~0.15 m tall, ~0.28 m wide)")
    ap.add_argument("--plant-hscale", type=float, default=4.5,
                    help="if >0, use this as the vertical (z) scale, keeping the "
                         "horizontal (x/y) scale at --plant-scale")
    ap.add_argument("--start-x", type=float, default=-9.0)
    ap.add_argument("--curve-amp", type=float, default=1.0,
                    help="lateral S-bend amplitude (m); 0 = straight rows. "
                         "Furrow center follows amp*sin(2*pi*(x-start_x)/period)")
    ap.add_argument("--curve-period", type=float, default=18.0,
                    help="S-bend wavelength along x (m)")
    ap.add_argument("--curve-phase", type=float, default=0.0,
                    help="S-bend phase (radians)")
    ap.add_argument("--curve-entry", type=float, default=6.0,
                    help="straight entry length (m) from start_x over which the "
                         "bend amplitude ramps 0->full (smoothstep), so the "
                         "rover spawns looking down a straight furrow and the "
                         "bends develop downfield")
    ap.add_argument("--shape", choices=("straight", "circle", "zigzag"), default="straight",
                    help="field shape: straight rows along +x (optional S-bend), "
                         "concentric circular rows around (--circle-cx, --circle-cy), "
                         "or a gentle two-tone zigzag imitating real planting wobble")
    ap.add_argument("--circle-r", type=float, default=8.0,
                    help="circle shape: furrow-center radius (m)")
    ap.add_argument("--circle-cx", type=float, default=0.0,
                    help="circle shape: center x (m)")
    ap.add_argument("--circle-cy", type=float, default=0.0,
                    help="circle shape: center y (m)")
    ap.add_argument("--direction", choices=("ccw", "cw"), default="ccw",
                    help="circle shape: travel direction (spawn yaw follows it)")
    ap.add_argument("--spawn-yaw-offset", type=float, default=0.0,
                    help="added to the circle tangent spawn yaw (rad, + = into "
                         "a CCW turn): look-into-the-corner lead so the initial "
                         "view centers the bending corridor")
    ap.add_argument("--zigzag-amp1", type=float, default=0.3,
                    help="zigzag shape: primary lateral amplitude (m)")
    ap.add_argument("--zigzag-period1", type=float, default=18.0,
                    help="zigzag shape: primary wavelength along x (m)")
    ap.add_argument("--zigzag-amp2", type=float, default=0.1,
                    help="zigzag shape: secondary lateral amplitude (m)")
    ap.add_argument("--zigzag-period2", type=float, default=8.0,
                    help="zigzag shape: secondary wavelength along x (m)")
    ap.add_argument("--lane-y", type=float, default=None,
                    help="furrow center to drive (m). Default: 0.0 for 2 rows, "
                         "spacing/2 (first gap right of middle) otherwise")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    out = Path(args.out) if args.out else here.parent / "worlds" / "farm_maize.world"
    out.parent.mkdir(parents=True, exist_ok=True)

    # symmetric row offsets around x axis: central gap of width `spacing`
    n_rows = args.rows
    offsets = [(i - (n_rows - 1) / 2.0) * args.spacing for i in range(n_rows)]
    lane_y = args.lane_y if args.lane_y is not None else \
        (0.0 if n_rows == 2 else args.spacing / 2.0)

    models_used = {"big_plant": 0, "small_plant": 0}
    plants = []
    random.seed(7)
    idx = 0
    two_pi = 2.0 * math.pi
    row_model = "big_plant"   # small_plant (5 cm) is too short for a 0.5 m camera
    spawn = {"robot_x": args.start_x + 1.0, "robot_y": lane_y, "robot_yaw": 0.0}
    lane = {"shape": args.shape, "circle_cx": 0.0, "circle_cy": 0.0,
            "circle_r": 0.0, "max_laps_default": 1,
            "vertical_coverage_default": 0.75}
    gui_pose = None
    if args.shape == "circle":
        if n_rows != 2:
            print("[WARN] circle shape supports 2 rows; got "
                  f"{n_rows}, using 2", file=sys.stderr)
            n_rows = 2
            offsets = [-args.spacing / 2.0, args.spacing / 2.0]
        R, cx, cy = args.circle_r, args.circle_cx, args.circle_cy
        for oy in offsets:
            rr = R + oy
            n = max(8, int(round(two_pi * rr / args.plant_spacing)))
            for k in range(n):
                ang = two_pi * k / n + random.gauss(0, args.noise * 0.15 / max(rr, 1.0))
                rad = rr + random.gauss(0, args.noise)
                x = cx + rad * math.cos(ang)
                y = cy + rad * math.sin(ang)
                plants.append((row_model, idx, x, y))
                models_used[row_model] += 1
                idx += 1
        # spawn on the furrow at angle 0, facing the tangent (CCW: +y, CW: -y)
        # plus --spawn-yaw-offset look-into-the-corner lead
        yaw = (math.pi / 2.0 if args.direction == "ccw" else -math.pi / 2.0) \
            + args.spawn_yaw_offset
        spawn = {"robot_x": cx + R, "robot_y": cy, "robot_yaw": yaw}
        # half-lap default: the P-controller slowly cuts inside on constant
        # curvature (alternating bends self-cancel, constant ones integrate),
        # so a full ring is beyond it; 180 deg of continuous bending is the
        # honest demo (pass --laps 0 / max_laps:=0 to loop until lost).
        lane = {"shape": "circle", "circle_cx": cx, "circle_cy": cy,
                "circle_r": R, "max_laps_default": 0.5,
                # shorter lookahead cuts inside less on constant curvature
                "vertical_coverage_default": 0.6}
        # GUI camera behind/above/side of spawn, looking down the tangent
        sdx, sdy = (-2.8, 3.4) if args.direction == "ccw" else (-2.8, -3.4)
        # rotate the straight-world offset pattern by the spawn yaw
        ox = sdx * math.cos(yaw) - sdy * math.sin(yaw)
        oy_ = sdx * math.sin(yaw) + sdy * math.cos(yaw)
        gui_pose = (spawn["robot_x"] + ox, spawn["robot_y"] + oy_, 4.6,
                    0.0, math.radians(38), yaw)
    else:
        n = int(args.length / args.plant_spacing)
        for oy in offsets:
            for k in range(n):
                x = args.start_x + k * args.plant_spacing + random.gauss(0, args.noise * 0.4)
                # Furrow-center lateral offset at x (S-bend and zigzag shift
                # all rows together). The amplitude ramps 0->full over
                # --curve-entry meters so the spawn area stays straight
                # (default spawn x/y/yaw keep working) and bends develop
                # downfield where the camera can see them coming.
                # S-bend: y_c = amp*sin(2*pi*(x-start_x)/period + phase);
                #   max heading ~= atan(amp*2*pi/period): amp=1, period=18
                #   -> ~19 deg, clearly visible but drivable.
                # Zigzag: two incommensurate sines (gentle real-world wobble,
                #   max slope only a few degrees, no sustained curvature).
                y_c = 0.0
                dx_e = x - args.start_x
                if args.curve_entry > 0 and dx_e < args.curve_entry:
                    s = max(0.0, min(1.0, dx_e / args.curve_entry))
                    env = s * s * (3.0 - 2.0 * s)
                else:
                    env = 1.0
                if args.shape == "zigzag":
                    y_c = env * (
                        args.zigzag_amp1 * math.sin(two_pi * dx_e / args.zigzag_period1)
                        + args.zigzag_amp2 * math.sin(two_pi * dx_e / args.zigzag_period2 + 1.3))
                elif args.curve_amp and args.curve_period > 0:
                    y_c = env * args.curve_amp * math.sin(
                        two_pi * (x - args.start_x) / args.curve_period + args.curve_phase)
                y = oy + y_c + random.gauss(0, args.noise)
                if abs(oy) < args.spacing:
                    # middle rows: leave a clean furrow band near the travel lane is fine
                    pass
                plants.append((row_model, idx, x, y))
                models_used[row_model] += 1
                idx += 1

    soil_r, soil_g, soil_b = 0.36, 0.26, 0.16   # brownish soil
    plant_scale = args.plant_scale
    if args.plant_hscale > 0:
        sx = sy = plant_scale
        sz = args.plant_hscale
    else:
        sx = sy = sz = plant_scale
    body = []
    body.append("<?xml version='1.0'?>\n")
    body.append("<sdf version='1.6'>\n")
    body.append("  <world name='farm_maize'>\n")
    body.append("    <light name='sun' type='directional'>\n")
    body.append("      <cast_shadows>1</cast_shadows>\n")
    body.append("      <pose>0 0 12 0 0 0</pose>\n")
    body.append("      <diffuse>0.85 0.85 0.8 1</diffuse>\n")
    body.append("      <specular>0.1 0.1 0.1 1</specular>\n")
    body.append("      <direction>-0.4 0.1 -1</direction>\n")
    body.append("    </light>\n")
    body.append("    <gravity>0 0 -9.8</gravity>\n")
    body.append("    <physics name='default_physics' type='ode'>\n")
    body.append("      <max_step_size>0.002</max_step_size>\n")
    body.append("      <real_time_factor>1</real_time_factor>\n")
    body.append("      <real_time_update_rate>500</real_time_update_rate>\n")
    body.append("    </physics>\n")
    body.append("    <scene><ambient>0.45 0.45 0.42 1</ambient>"
                "<background>0.75 0.75 0.72 1</background><shadows>1</shadows></scene>\n")
    # GUI camera: start the gzclient view just behind/above the rover start,
    # looking down the furrow. Users can orbit/zoom freely afterwards.
    if gui_pose is None:
        gui_pose = (args.start_x - 2.8, 3.4, 4.6, 0.0, math.radians(38), 0.0)
    body.append("    <gui>\n")
    body.append("      <camera name='user_camera'>\n")
    body.append(f"        <pose>{fmt(gui_pose[0])} {fmt(gui_pose[1])} {fmt(gui_pose[2])} "
                f"{fmt(gui_pose[3])} {fmt(gui_pose[4])} {fmt(gui_pose[5])}</pose>\n")
    body.append("      </camera>\n")
    body.append("    </gui>\n")
    # soil ground plane
    body.append("    <model name='ground_plane'>\n")
    body.append("      <static>1</static>\n")
    body.append("      <link name='link'>\n")
    body.append("        <collision name='collision'>\n")
    body.append("          <geometry><plane><normal>0 0 1</normal>"
                "<size>80 40</size></plane></geometry>\n")
    body.append("          <surface><friction><ode><mu>0.9</mu><mu2>0.8</mu2></ode>"
                "</friction></surface>\n")
    body.append("          <max_contacts>10</max_contacts>\n")
    body.append("        </collision>\n")
    body.append("        <visual name='visual'>\n")
    body.append("          <cast_shadows>0</cast_shadows>\n")
    body.append("          <geometry><plane><normal>0 0 1</normal>"
                "<size>80 40</size></plane></geometry>\n")
    body.append("          <material>\n")
    body.append(f"            <ambient>{soil_r} {soil_g} {soil_b} 1</ambient>\n")
    body.append(f"            <diffuse>{soil_r} {soil_g} {soil_b} 1</diffuse>\n")
    body.append(f"            <specular>0.02 0.02 0.02 1</specular>\n")
    body.append("          </material>\n")
    body.append("        </visual>\n")
    body.append("      </link>\n")
    body.append("    </model>\n")
    body.append("    <model name='start_gate'>\n")
    body.append("      <static>1</static>\n")
    body.append(f"      <pose>{fmt(spawn['robot_x'])} {fmt(spawn['robot_y'])} 0 0 0 0</pose>\n")
    body.append("      <link name='link'>\n")
    body.append("        <visual name='v'>\n")
    body.append("          <pose>0 0 0.05 0 0 0</pose>\n")
    body.append("          <geometry><box><size>0.05 0.5 0.1</size></box></geometry>\n")
    body.append("          <material><diffuse>1 0 0 1</diffuse></material>\n")
    body.append("        </visual>\n")
    body.append("      </link>\n")
    body.append("    </model>\n")
    # plants: visual mesh + a thin static stem collision (cheap cylinder).
    # The visual mesh alone is invisible to ray sensors (no <collision>),
    # which left the side ToF rangers blind; the stem (r=5 cm, z 0..0.6 m)
    # makes plants ray-visible. collide_without_contact keeps it
    # sensor-only: the rover passes through stems with zero contact force
    # (no tripping/toppling - toppling is not a real-life scenario here),
    # while the ToF guard still sees the rows and steers away. The wide
    # leaf canopy stays visual-only so normal tracking never snags. The
    # agribot big_plant STL is only ~0.15 m tall; scale it up so rows read
    # clearly from the rover camera (~0.5 m high).
    for m, i, x, y in plants:
        body.append(
            f"    <model name='pl_{i}'>\n"
            f"      <static>1</static>\n"
            f"      <pose>{fmt(x)} {fmt(y)} 0 0 0 0</pose>\n"
            f"      <link name='l'>\n"
            f"        <collision name='stem'>\n"
            f"          <pose>0 0 0.30 0 0 0</pose>\n"
            f"          <geometry><cylinder><radius>0.05</radius>"
            f"<length>0.60</length></cylinder></geometry>\n"
            f"          <surface><friction><ode><mu>0.9</mu><mu2>0.8</mu2></ode>"
            f"</friction><contact><collide_without_contact>true"
            f"</collide_without_contact></contact></surface>\n"
            f"          <max_contacts>4</max_contacts>\n"
            f"        </collision>\n"
            f"        <visual name='v'>\n"
            f"          <geometry>\n"
            f"            <mesh>\n"
            f"              <uri>model://big_plant/mesh/big_plant.stl</uri>\n"
            f"              <scale>{fmt(sx)} {fmt(sy)} {fmt(sz)}</scale>\n"
            f"            </mesh>\n"
            f"          </geometry>\n"
            f"          <material><ambient>0.15 0.65 0.2 1</ambient>"
            f"<diffuse>0.15 0.65 0.2 1</diffuse></material>\n"
            f"        </visual>\n"
            f"      </link>\n"
            f"    </model>\n")
    body.append("  </world>\n")
    body.append("</sdf>\n")

    out.write_text("".join(body))
    sidecar = out.parent / (out.stem + ".spawn.json")
    # furrow centers at the field start (entry ramp ~0 there): gap i sits
    # halfway between rows i and i+1, used for --spawn N lane selection.
    furrows = sorted([(offsets[i] + offsets[i + 1]) / 2.0
                      for i in range(len(offsets) - 1)]) \
        if args.shape != "circle" else [0.0]
    sidecar.write_text(json.dumps(
        {"world": out.name, "shape": lane["shape"],
         "robot_x": spawn["robot_x"], "robot_y": spawn["robot_y"],
         "robot_yaw": spawn["robot_yaw"],
         "lane_y": float(spawn["robot_y"]), "lane_end_x": 9.0,
         "n_rows": n_rows, "row_spacing": args.spacing,
         "furrows": [round(float(c), 3) for c in furrows],
          "circle_cx": lane["circle_cx"], "circle_cy": lane["circle_cy"],
          "circle_r": lane["circle_r"],
          "max_laps_default": lane["max_laps_default"],
          "vertical_coverage_default": lane["vertical_coverage_default"]}, indent=2) + "\n")
    print(f"wrote {out}")
    print(f"wrote {sidecar}")
    print(f"rows: {n_rows} at offsets {[round(o,2) for o in offsets]} m, {len(plants)} plants "
          f"({dict(models_used)}), plant step {args.plant_spacing} m, "
          f"plant scale {sx}x{sz} (~{0.148*sz:.2f} m tall, ~{0.28*sx:.2f} m wide)")
    if args.shape == "circle":
        print(f"circle: R={args.circle_r} m around ({args.circle_cx}, {args.circle_cy}), "
              f"{args.direction}; spawn=({spawn['robot_x']:.2f}, {spawn['robot_y']:.2f}, "
              f"yaw={spawn['robot_yaw']:.2f})")
    elif args.shape == "zigzag":
        print(f"zigzag: {args.zigzag_amp1} m / {args.zigzag_period1} m + "
              f"{args.zigzag_amp2} m / {args.zigzag_period2} m; "
              f"lane_y={lane_y:.2f}")
    elif args.curve_amp:
        print(f"curve: amp {args.curve_amp} m, period {args.curve_period} m, "
              f"phase {args.curve_phase} rad (S-bend furrow, max slope "
              f"~{math.degrees(math.atan(args.curve_amp * 2 * math.pi / args.curve_period)):.0f} deg)")
    else:
        print("curve: straight rows")

if __name__ == "__main__":
    main()
