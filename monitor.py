"""
monitor.py - Real measurements from the machine running the app.

What this CAN measure (on your own computer):
  * latency + packet loss to one target host, via the system `ping` command
  * jitter (variation in ping times)
  * throughput, interface errors/drops, interface up/down - via psutil
  * CPU / memory usage - via psutil

What it CANNOT measure: router CPU, SNMP counters, BGP/OSPF state, links
inside an ISP network, etc.  So a model trained on ISP router data (like
LCORE-D) generally cannot be fed by this monitor - the columns don't match.
The API tells you which features are missing instead of making them up.

If `ping` is not installed (or not allowed), we fall back to timing TCP
connections. That is labelled `tcp_connect`; its failure rate is NOT ICMP
packet loss, so packet_loss_pct is left empty in that case.
"""
import platform
import re
import shutil
import socket
import statistics
import subprocess
import time
from datetime import datetime, timezone

import psutil

HOST_RE = re.compile(r"^[A-Za-z0-9.\-:]{1,253}$")


def valid_host(host):
    """Allow only hostname / IPv4 / IPv6 characters (blocks option injection like '-f')."""
    return bool(host) and HOST_RE.match(host) is not None and not host.startswith("-")


def _ping(host, count, timeout_s):
    system = platform.system().lower()
    if system == "windows":
        cmd = ["ping", "-n", str(count), "-w", str(int(timeout_s * 1000)), host]
    elif system == "darwin":
        cmd = ["ping", "-c", str(count), "-W", str(int(timeout_s * 1000)), host]
    else:
        cmd = ["ping", "-c", str(count), "-W", str(int(timeout_s)), "-i", "0.3", host]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=count * (timeout_s + 1) + 5)
    out = proc.stdout + proc.stderr

    times = [float(t) for t in re.findall(r"time[=<]\s*([\d.]+)\s*ms", out)]
    loss = re.search(r"([\d.]+)%\s*(packet\s*)?loss", out)
    if loss is None:
        sent = re.search(r"Sent = (\d+), Received = (\d+)", out)  # Windows
        loss_pct = (100.0 * (1 - int(sent.group(2)) / int(sent.group(1)))) if sent else None
    else:
        loss_pct = float(loss.group(1))
    if loss_pct is None and not times:
        raise RuntimeError(out.strip()[:200] or "ping produced no output")
    return {
        "method": "icmp_ping",
        "latency_ms": round(statistics.mean(times), 3) if times else None,
        "jitter_ms": round(statistics.pstdev(times), 3) if len(times) > 1 else None,
        "packet_loss_pct": loss_pct,
        "probe_failure_pct": loss_pct,
        "probes": count,
    }


def _tcp_probe(host, count, timeout_s, port=443):
    times, failures = [], 0
    for _ in range(count):
        start = time.perf_counter()
        try:
            with socket.create_connection((host, port), timeout=timeout_s):
                times.append((time.perf_counter() - start) * 1000)
        except OSError:
            failures += 1
    return {
        "method": f"tcp_connect:{port}",
        "latency_ms": round(statistics.mean(times), 3) if times else None,
        "jitter_ms": round(statistics.pstdev(times), 3) if len(times) > 1 else None,
        "packet_loss_pct": None,  # TCP connect failures are NOT packet loss
        "probe_failure_pct": round(100.0 * failures / count, 2),
        "probes": count,
    }


def measure_reachability(host, count=4, timeout_s=2.0):
    """Ping if possible, else TCP-connect timing. Never invents numbers."""
    errors = []
    if shutil.which("ping"):
        try:
            return _ping(host, count, timeout_s)
        except Exception as exc:  # noqa: BLE001 - report, then fall back
            errors.append(f"ping failed: {exc}")
    else:
        errors.append("ping command not available")
    result = _tcp_probe(host, count, timeout_s)
    result["notes"] = errors
    return result


def measure_interfaces(interval_s=1.0):
    """Sample psutil counters twice and turn the difference into per-second rates."""
    a = psutil.net_io_counters()
    time.sleep(interval_s)
    b = psutil.net_io_counters()
    d = lambda attr: max(0, getattr(b, attr) - getattr(a, attr)) / interval_s  # noqa: E731
    stats = psutil.net_if_stats()
    non_loopback = {n: s for n, s in stats.items() if not n.lower().startswith(("lo", "loopback"))}
    return {
        "bytes_sent_per_s": round(d("bytes_sent"), 1),
        "bytes_recv_per_s": round(d("bytes_recv"), 1),
        "throughput_mbps": round((d("bytes_sent") + d("bytes_recv")) * 8 / 1e6, 4),
        "packets_sent_per_s": round(d("packets_sent"), 1),
        "packets_recv_per_s": round(d("packets_recv"), 1),
        "interface_errors_per_s": round(d("errin") + d("errout"), 3),
        "interface_drops_per_s": round(d("dropin") + d("dropout"), 3),
        "interfaces_up": sum(1 for s in non_loopback.values() if s.isup),
        "interfaces_total": len(non_loopback),
    }


def collect(host="8.8.8.8", count=4):
    if not valid_host(host):
        raise ValueError(f"invalid host: {host!r}")
    data = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "host": host}
    data.update(measure_interfaces())
    data.update(measure_reachability(host, count=count))
    data["cpu_percent"] = psutil.cpu_percent(interval=None)
    data["memory_percent"] = psutil.virtual_memory().percent
    return data


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(collect(sys.argv[1] if len(sys.argv) > 1 else "8.8.8.8"), indent=2))
