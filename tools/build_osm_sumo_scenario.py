#!/usr/bin/env python3
"""Build a SUMO scenario from an exported OSM file.

This helper is designed for the Central Ave EB through-only transfer scenario.
It converts an OSM extract (.osm or .osm.xml) to a SUMO network, asks duarouter for one route
between the selected west and east edges, and writes a route template that the
existing environment can later rescale by total_flow_vph and CAV penetration.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_indices(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(x) for x in value.replace(",", " ").split()]


def parse_strings(value: str | None) -> list[str]:
    if not value:
        return []
    return [x for x in value.replace(",", " ").split() if x]


def write_xml(path: Path, root: ET.Element) -> None:
    ET.indent(root, space="    ")
    ET.ElementTree(root).write(path, encoding="UTF-8", xml_declaration=True)


def build_net(osm_file: Path, net_file: Path) -> None:
    run([
        "netconvert",
        "--osm-files", str(osm_file),
        "--output-file", str(net_file),
        "--geometry.remove",
        "--junctions.join",
        "--tls.guess",
        "--tls.guess-signals",
        "--tls.join",
        "--no-turnarounds", "true",
    ])


def apply_network_patch(
    net_file: Path,
    node_file: Path | None,
    edge_file: Path | None,
    connection_file: Path | None,
    remove_edges: list[str],
) -> None:
    patched_file = net_file.with_name(f"{net_file.stem}.patched.xml")
    cmd = [
        "netconvert",
        "--sumo-net-file", str(net_file),
        "--output-file", str(patched_file),
        "--no-turnarounds", "true",
    ]
    if node_file is not None:
        cmd += ["--node-files", str(node_file)]
    if edge_file is not None:
        cmd += ["--edge-files", str(edge_file)]
    if connection_file is not None:
        cmd += ["--connection-files", str(connection_file)]
    if remove_edges:
        cmd += ["--remove-edges.explicit", ",".join(remove_edges)]
    run(cmd)
    patched_file.replace(net_file)


def extract_route_edges(net_file: Path, from_edge: str, to_edge: str, work_dir: Path) -> str:
    trips_file = work_dir / "_probe_trips.xml"
    routed_file = work_dir / "_probe_routes.rou.xml"
    root = ET.Element("routes")
    ET.SubElement(
        root,
        "trip",
        id="probe",
        depart="0",
        **{"from": from_edge, "to": to_edge},
    )
    write_xml(trips_file, root)
    run([
        "duarouter",
        "--net-file", str(net_file),
        "--route-files", str(trips_file),
        "--output-file", str(routed_file),
        "--ignore-errors", "true",
    ])
    routed_root = ET.parse(routed_file).getroot()
    route = routed_root.find(".//route")
    if route is None or not route.get("edges"):
        raise RuntimeError(
            "duarouter did not find a route. Recheck --from-edge and --to-edge."
        )
    return route.get("edges", "")


def write_route_template(
    route_file: Path,
    route_edges: str,
    demand_vph: float,
    cav_penetration: float,
    begin_s: float,
    end_s: float,
    depart_speed: float,
    depart_lanes: list[str],
) -> None:
    cav_flow = demand_vph * cav_penetration
    hdv_flow = demand_vph - cav_flow
    root = ET.Element("routes")
    ET.SubElement(
        root,
        "vType",
        id="hdv",
        accel="3.0",
        decel="5.0",
        sigma="0.0",
        length="5.0",
        tau="1.5",
        minGap="2.0",
        maxSpeed="18.00",
        color="1,1,0",
        lcKeepRight="0",
        speedFactor="1.0",
    )
    ET.SubElement(
        root,
        "vType",
        id="cav",
        accel="3.0",
        decel="5.0",
        sigma="0.0",
        length="5.0",
        tau="1.5",
        minGap="2.0",
        maxSpeed="18.00",
        color="0,1,0",
        lcKeepRight="0",
        speedFactor="1.0",
    )
    ET.SubElement(root, "route", id="central_eb_through", edges=route_edges)
    flow_common = {
        "route": "central_eb_through",
        "begin": f"{begin_s:g}",
        "end": f"{end_s:g}",
        "departSpeed": f"{depart_speed:g}",
    }
    if depart_lanes:
        lane_count = len(depart_lanes)
        for lane in depart_lanes:
            lane_tag = lane.replace("-", "m").replace(".", "p")
            ET.SubElement(
                root,
                "flow",
                id=f"hdv_central_eb_lane{lane_tag}",
                type="hdv",
                vehsPerHour=f"{hdv_flow / lane_count:.6f}",
                departLane=lane,
                **flow_common,
            )
            ET.SubElement(
                root,
                "flow",
                id=f"cav_central_eb_lane{lane_tag}",
                type="cav",
                vehsPerHour=f"{cav_flow / lane_count:.6f}",
                departLane=lane,
                **flow_common,
            )
    else:
        ET.SubElement(
            root,
            "flow",
            id="hdv_central_eb",
            type="hdv",
            vehsPerHour=f"{hdv_flow:.6f}",
            departLane="random",
            **flow_common,
        )
        ET.SubElement(
            root,
            "flow",
            id="cav_central_eb",
            type="cav",
            vehsPerHour=f"{cav_flow:.6f}",
            departLane="random",
            **flow_common,
        )
    write_xml(route_file, root)


def find_tls(net_file: Path, tls_id: str | None) -> tuple[str | None, int]:
    root = ET.parse(net_file).getroot()
    tl_logics = root.findall("tlLogic")
    if tls_id:
        tl_logics = [tl for tl in tl_logics if tl.get("id") == tls_id]
    if not tl_logics:
        return None, 0
    tl = tl_logics[0]
    phase = tl.find("phase")
    if phase is None or not phase.get("state"):
        return tl.get("id"), 0
    return tl.get("id"), len(phase.get("state", ""))


def write_tls_file(
    tll_file: Path,
    tls_id: str,
    state_len: int,
    eb_link_indices: list[int],
    eb_minor_link_indices: list[int],
    green_s: float,
    yellow_s: float,
    red_s: float,
) -> None:
    if state_len <= 0:
        raise RuntimeError("Cannot infer traffic-light state length from the net file.")
    green_state = ["r"] * state_len
    yellow_state = ["r"] * state_len
    for idx in eb_link_indices:
        if idx < 0 or idx >= state_len:
            raise ValueError(f"EB link index {idx} outside valid range 0..{state_len - 1}.")
        green_state[idx] = "G"
        yellow_state[idx] = "y"
    for idx in eb_minor_link_indices:
        if idx < 0 or idx >= state_len:
            raise ValueError(f"EB minor link index {idx} outside valid range 0..{state_len - 1}.")
        green_state[idx] = "g"
        yellow_state[idx] = "y"
    root = ET.Element("additional")
    tl = ET.SubElement(root, "tlLogic", id=tls_id, type="static", programID="pm2026", offset="0")
    ET.SubElement(tl, "phase", duration=f"{green_s:g}", state="".join(green_state))
    ET.SubElement(tl, "phase", duration=f"{yellow_s:g}", state="".join(yellow_state))
    ET.SubElement(tl, "phase", duration=f"{red_s:g}", state="r" * state_len)
    write_xml(tll_file, root)


def write_sumocfg(sumocfg: Path, net_file: Path, route_file: Path, tll_file: Path | None) -> None:
    root = ET.Element("configuration")
    input_el = ET.SubElement(root, "input")
    ET.SubElement(input_el, "net-file", value=net_file.name)
    ET.SubElement(input_el, "route-files", value=route_file.name)
    if tll_file is not None:
        ET.SubElement(input_el, "additional-files", value=tll_file.name)
    time_el = ET.SubElement(root, "time")
    ET.SubElement(time_el, "begin", value="0")
    ET.SubElement(time_el, "end", value="3600")
    ET.SubElement(time_el, "step-length", value="0.2")
    write_xml(sumocfg, root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--osm", required=True, type=Path, help="Exported .osm or .osm.xml file.")
    parser.add_argument("--out-dir", default=Path("envs/sumo_files/sycamore_central_pm2026"), type=Path)
    parser.add_argument("--name", default="sycamore_central_pm2026")
    parser.add_argument("--from-edge", required=True, help="Central Ave EB upstream/west edge id.")
    parser.add_argument("--to-edge", required=True, help="Central Ave EB downstream/east edge id.")
    parser.add_argument("--extra-node-file", default=None, type=Path, help="Optional SUMO node patch applied after OSM conversion.")
    parser.add_argument("--extra-edge-file", default=None, type=Path, help="Optional SUMO edge patch applied after OSM conversion.")
    parser.add_argument("--extra-connection-file", default=None, type=Path, help="Optional SUMO connection patch applied after OSM conversion.")
    parser.add_argument("--remove-edges", default=None, help="Comma/space-separated edge ids to remove after OSM conversion.")
    parser.add_argument("--controlled-edge", default=None, help="Edge id where RL controls CAVs. Defaults to --from-edge.")
    parser.add_argument("--tls-id", default=None, help="Traffic light id. Defaults to the first tlLogic in net.")
    parser.add_argument("--eb-link-indices", default=None, help="Comma/space-separated TLS link indices for EB through lanes.")
    parser.add_argument("--eb-minor-link-indices", default=None, help="EB through link indices that should use permissive green 'g'.")
    parser.add_argument("--demand-vph", type=float, default=739.0, help="2026 PM EBT demand.")
    parser.add_argument("--initial-cav-penetration", type=float, default=0.5)
    parser.add_argument("--depart-lanes", default=None, help="Comma/space-separated depart lanes. Example: '1,2,3'. Defaults to random lanes.")
    parser.add_argument("--green-s", type=float, default=17.0)
    parser.add_argument("--yellow-s", type=float, default=5.0)
    parser.add_argument("--red-s", type=float, default=68.0)
    parser.add_argument("--begin-s", type=float, default=0.0)
    parser.add_argument("--end-s", type=float, default=360.0)
    parser.add_argument("--depart-speed", type=float, default=10.0)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    net_file = out_dir / f"{args.name}.net.xml"
    route_file = out_dir / f"{args.name}.rou.xml"
    tll_file = out_dir / f"{args.name}.tll.xml"
    sumocfg = out_dir / f"{args.name}.sumocfg"

    build_net(args.osm, net_file)
    if (
        args.extra_node_file is not None
        or args.extra_edge_file is not None
        or args.extra_connection_file is not None
        or args.remove_edges is not None
    ):
        apply_network_patch(
            net_file,
            args.extra_node_file,
            args.extra_edge_file,
            args.extra_connection_file,
            parse_strings(args.remove_edges),
        )
    route_edges = extract_route_edges(net_file, args.from_edge, args.to_edge, out_dir)
    write_route_template(
        route_file,
        route_edges,
        args.demand_vph,
        args.initial_cav_penetration,
        args.begin_s,
        args.end_s,
        args.depart_speed,
        parse_strings(args.depart_lanes),
    )

    final_tll = None
    tls_id, state_len = find_tls(net_file, args.tls_id)
    eb_link_indices = parse_indices(args.eb_link_indices)
    eb_minor_link_indices = parse_indices(args.eb_minor_link_indices)
    if eb_link_indices:
        if tls_id is None:
            raise RuntimeError("No tlLogic found in the generated net file.")
        write_tls_file(
            tll_file,
            tls_id,
            state_len,
            eb_link_indices,
            eb_minor_link_indices,
            args.green_s,
            args.yellow_s,
            args.red_s,
        )
        final_tll = tll_file
    write_sumocfg(sumocfg, net_file, route_file, final_tll)

    controlled_edge = args.controlled_edge or args.from_edge
    print("\nBuilt scenario:")
    print(f"  sumo_cfg: {sumocfg}")
    print(f"  route_template: {route_file}")
    print(f"  controlled_edges: {controlled_edge}")
    print(f"  route edges: {route_edges}")
    if final_tll is None:
        print("  traffic light: kept netconvert-generated timing; rerun with --eb-link-indices to override PM timing.")
    else:
        print(f"  traffic light override: {final_tll}")
        print(f"  terminal green window for training: 0.0 to {args.green_s + args.yellow_s:g} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
