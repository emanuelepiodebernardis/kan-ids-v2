#!/usr/bin/env python3
"""Build, upload and coordinate one Mega/FNB58 pilot; never calculate energy.

Only this extracted package is built. The external UsbMeter application owns
the FNB58. Host receipt timestamps are transport timestamps, not GPIO times.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid
import zipfile

ENV = "mega_fnb58_coeff_pilot"
PROTOCOL = "kanids-mega-fnb58-pilot-v1"
COHORT_SHA = "20c53b571bfaef91f5fd26b19e5960e3fa85ed48b38c78c8527245b921ebe86f"
MEGA_SERIAL = "14532303532351804271"
RAW_IDS = [57978, 126454, 49019, 94298, 183174, 127803, 12386, 91098,
           14299, 116321, 148681, 119000, 150843, 98347, 80922, 124992,
           204398, 82285, 25406, 123258]
EVENT_NAMES = ["pre_led_on_1", "pre_led_off_1", "pre_led_on_2",
               "pre_led_off_2", "pre_led_on_3", "pre_led_off_3",
               "active_begin", "active_end", "post_led_on_1",
               "post_led_off_1", "post_led_on_2", "post_led_off_2",
               "post_led_on_3", "post_led_off_3"]


class PilotError(RuntimeError):
    pass


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def digest(path: Path) -> dict:
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def verify_sources(root: Path) -> list[dict]:
    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise PilotError("Пустой или неверный SOURCE_MANIFEST.json")
    seen = set()
    for entry in entries:
        name = entry.get("path", "")
        path = (root / name).resolve()
        if not name or name in seen or not path.is_relative_to(root.resolve()):
            raise PilotError("Неверный путь в манифесте: " + str(name))
        seen.add(name)
        if not path.is_file() or digest(path) != {"bytes": entry.get("bytes"),
                                                "sha256": entry.get("sha256")}:
            raise PilotError("Файл не совпадает с манифестом: " + name)
    required = {"project/platformio.ini", "project/src/main.cpp", "run_mega_energy_pilot.py"}
    required.update(p.relative_to(root).as_posix()
                    for p in (root / "project/include").rglob("*") if p.is_file())
    if not required.issubset(seen):
        raise PilotError("Исходный файл отсутствует в манифесте: " + str(required - seen))
    return entries


def check_board(ports, requested: str) -> dict:
    matches = [p for p in ports if p.device.casefold() == requested.casefold()]
    if len(matches) != 1:
        raise PilotError("Порт " + requested + " не найден однозначно; прошивка не начата.")
    p = matches[0]
    if (p.vid, p.pid, p.serial_number) != (0x2341, 0x0042, MEGA_SERIAL):
        raise PilotError("На " + requested + " другая плата; прошивка не начата.")
    return {"port": p.device, "vid": p.vid, "pid": p.pid,
            "serial_number": p.serial_number, "description": p.description}


def fields(line: str, tag: str) -> dict:
    parts = line.split()
    if not parts or parts[0] != tag:
        raise PilotError("Ожидалось " + tag + ", получено: " + line)
    values = {}
    for part in parts[1:]:
        if "=" not in part:
            raise PilotError("Некорректная строка протокола: " + line)
        key, value = part.split("=", 1)
        if not key or not value or key in values:
            raise PilotError("Повторное/пустое поле: " + line)
        values[key] = value
    return values


def integer(values: dict, name: str) -> int:
    value = values.get(name, "")
    if not re.fullmatch(r"[0-9]+", value):
        raise PilotError("Нет целого поля " + name)
    return int(value)


def validate_ready(lines: list[str]) -> dict:
    tags = {}
    for line in lines:
        tag = line.split()[0] if line.split() else ""
        if tag in {"ERROR", "EXECUTION", "DONE", "EVENT", "WAIT_DONE"}:
            raise PilotError("Недопустимое состояние до RUN: " + line)
        if tag in {"HELLO", "MODEL", "COHORT", "RAW_IDS", "CORRECTNESS", "READY"}:
            if tag in tags:
                raise PilotError("Перезапуск или повтор заголовка: " + tag)
            tags[tag] = fields(line, tag)
    if set(tags) != {"HELLO", "MODEL", "COHORT", "RAW_IDS", "CORRECTNESS", "READY"}:
        raise PilotError("Неполный заголовок прошивки")
    if tags["HELLO"] != {"protocol": PROTOCOL}:
        raise PilotError("Не та версия протокола")
    if tags["MODEL"] != {"variant": "coeff_int8", "model_bytes": "254", "cohort_sha256": COHORT_SHA}:
        raise PilotError("Модель или когорта не совпадает")
    if tags["COHORT"] != {"rows": "first20", "order": "attack_normal_interleaved",
                          "boundary": "prepared_features_in_RAM"}:
        raise PilotError("Другая выборка или граница измерения")
    if tags["RAW_IDS"] != {"values": ",".join(map(str, RAW_IDS))}:
        raise PilotError("Порядок исходных строк не совпадает")
    if tags["CORRECTNESS"] != {"checked": "20", "correct": "1", "reference": "compiled_C_predictions"}:
        raise PilotError("Индивидуальные проверки предсказаний не пройдены")
    ready = {k: integer(tags["READY"], k)
             for k in ("count", "target_us", "cal_n", "cal_us", "checksum_per20")}
    if not (ready["target_us"] == 120000000 and ready["cal_n"] == 200
            and ready["checksum_per20"] == 10 and 1000 <= ready["cal_us"] <= 30000000
            and 20 <= ready["count"] <= 5000000 and ready["count"] % 20 == 0):
        raise PilotError("Калибровка/число вычислений вне протокола")
    expected_n = (120000000 * 200 // ready["cal_us"] // 20) * 20
    if ready["count"] != expected_n:
        raise PilotError("Число вычислений не соответствует калибровке")
    return ready


def validate_done(lines: list[str], ready: dict) -> dict:
    done = None
    events = []
    prep = False
    terminal = False
    for line in lines:
        if terminal:
            raise PilotError("Данные после терминального сообщения")
        tag = line.split()[0] if line.split() else ""
        if tag == "PREP":
            if prep or done or events or fields(line, tag) != {"sync": "begin"}:
                raise PilotError("Некорректное начало опыта")
            prep = True
        elif tag == "DONE":
            if not prep or done is not None or events:
                raise PilotError("Повторное или преждевременное DONE")
            obj = fields(line, tag)
            done = {k: integer(obj, k) for k in
                    ("count", "active_us", "checksum", "expected", "correct", "timing_ok")}
        elif tag == "EVENT":
            if done is None:
                raise PilotError("EVENT до DONE")
            obj = fields(line, tag)
            events.append({"name": obj.get("name"), "us": integer(obj, "us")})
        elif tag == "WAIT_DONE":
            if fields(line, tag) != {"reset_required": "1"}:
                raise PilotError("Некорректное завершение")
            terminal = True
        else:
            raise PilotError("Неожиданный ответ во время опыта: " + line)
    if done is None or not terminal:
        raise PilotError("Опыт не завершён")
    expected = ready["count"] // 20 * ready["checksum_per20"]
    if not (done["count"] == ready["count"] and done["checksum"] == done["expected"] == expected
            and done["correct"] == done["timing_ok"] == 1
            and 60000000 <= done["active_us"] <= 180000000):
        raise PilotError("DONE: число, контрольная сумма или время не прошли проверку")
    if [e["name"] for e in events] != EVENT_NAMES:
        raise PilotError("Порядок/число событий не совпадает с протоколом")
    ts = [e["us"] for e in events]
    if not (0 <= ts[0] <= 100000 and all(b > a for a, b in zip(ts, ts[1:]))):
        raise PilotError("Немонотонные времена событий")
    if ts[7] - ts[6] != done["active_us"]:
        raise PilotError("События расходятся с active_us")
    expected_gaps = [2000000, 1000000, 4000000, 1000000, 2000000,
                     10000000, done["active_us"], 10000000,
                     2000000, 1000000, 4000000, 1000000, 2000000]
    for delta, expected_gap in zip((b - a for a, b in zip(ts, ts[1:])), expected_gaps):
        if abs(delta - expected_gap) > 100000:
            raise PilotError("Длительность маркера/защитного интервала вне допуска 100 ms")
    return {"done": done, "events": events, "software_protocol_pass": True,
            "external_trace_pending": True, "energy_result_available": False}


def run_command(command: list[str], run_dir: Path, stem: str, record: dict) -> None:
    info = {"argv": command, "started_utc": utc(), "log": stem + ".log"}
    record["commands"].append(info)
    print("\n> " + subprocess.list2cmdline(command), flush=True)
    with (run_dir / (stem + ".log")).open("w", encoding="utf-8", newline="\n") as handle:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        try:
            for line in proc.stdout:
                handle.write(line)
                handle.flush()
                print(line, end="", flush=True)
            code = proc.wait()
        except BaseException:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            info["interrupted"] = True
            raise
        finally:
            proc.stdout.close()
    info.update({"returncode": code, "finished_utc": utc()})
    if code:
        raise PilotError("Команда завершилась с ошибкой; следующий этап не запущен: " + stem)


class SerialCapture:
    def __init__(self, device, run_dir: Path):
        self.device = device
        self.raw = (run_dir / "serial_raw.bin").open("wb")
        self.log = (run_dir / "serial.jsonl").open("w", encoding="utf-8", newline="\n")
        self.text = (run_dir / "serial.log").open("w", encoding="utf-8", newline="\n")
        self.buffer = bytearray()
        self.pending = deque()

    def entry(self, direction: str, data: bytes) -> None:
        self.log.write(json.dumps({"utc": utc(), "host_monotonic_ns": time.perf_counter_ns(),
                                   "direction": direction, "hex": data.hex()}, ensure_ascii=True) + "\n")
        self.log.flush()

    def line(self, deadline: float) -> str:
        while time.monotonic() < deadline:
            if self.pending:
                line = self.pending.popleft().decode("ascii", errors="strict").rstrip("\r")
                if line:
                    self.text.write(line + "\n")
                    self.text.flush()
                    print(line, flush=True)
                    return line
                continue
            data = self.device.read(min(max(self.device.in_waiting, 1), 4096))
            if not data:
                continue
            self.raw.write(data)
            self.raw.flush()
            self.entry("RX", data)
            self.buffer.extend(data)
            while b"\n" in self.buffer:
                value, _, rest = self.buffer.partition(b"\n")
                self.pending.append(bytes(value))
                self.buffer = bytearray(rest)
            if len(self.buffer) > 4096:
                raise PilotError("Слишком длинная строка UART")
        raise PilotError("Истекло время ожидания прошивки; измерение не принято")

    def send_run(self) -> None:
        self.entry("TX", b"RUN\n")
        if self.device.write(b"RUN\n") != 4:
            raise PilotError("Команда RUN передана не полностью")
        self.device.flush()

    def close(self) -> None:
        self.raw.close()
        self.log.close()
        self.text.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run", action="store_true", help="build, upload, then coordinate one pilot")
    action.add_argument("--build-only", action="store_true", help="build without connecting to the board")
    parser.add_argument("--port", default="COM5")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    run_id = datetime.now(timezone.utc).strftime("MEGA_PILOT_%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True)
    record = {"run_id": run_id, "started_utc": utc(), "status": "incomplete",
              "mode": "run" if args.run else "build_only", "commands": [],
              "python": sys.version, "system": platform.platform(), "requested_port": args.port,
              "scope": "USB measurement point includes whole board and downstream cable losses; prepared features; external V/I trace required",
              "host_timestamps": "transport receipt timestamps, not hardware timestamps",
              "energy_result_available": False}
    device = capture = None
    code = 1
    try:
        if sys.version_info[:2] != (3, 11):
            raise PilotError("Запустите через py -3.11")
        entries = verify_sources(root)
        record["source_manifest"] = digest(root / "SOURCE_MANIFEST.json")
        for entry in entries + [{"path": "SOURCE_MANIFEST.json"}]:
            src = root / entry["path"]
            dst = run_dir / "sources" / entry["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        pio_version = importlib.metadata.version("platformio")
        if pio_version != "6.1.19":
            raise PilotError("Требуется PlatformIO 6.1.19, установлен " + pio_version)
        import serial
        from serial.tools import list_ports
        record["versions"] = {"platformio": pio_version, "pyserial": importlib.metadata.version("pyserial")}
        if args.run:
            record["board_before_build"] = check_board(list_ports.comports(), args.port)
        project = root / "project"
        base = [sys.executable, "-m", "platformio"]
        run_command(base + ["run", "-d", str(project), "-e", ENV], run_dir, "01_build", record)
        run_command(base + ["pkg", "list", "-d", str(project), "-e", ENV], run_dir, "02_packages", record)
        verify_sources(root)
        artifacts = {}
        for name in ("firmware.elf", "firmware.hex"):
            path = project / ".pio" / "build" / ENV / name
            if not path.is_file():
                raise PilotError("Не найден результат сборки: " + name)
            artifacts[name] = digest(path)
            shutil.copy2(path, run_dir / name)
        record["binaries"] = artifacts
        if args.build_only:
            record["status"] = "build_pass_no_upload"
            code = 0
        else:
            record["board_before_upload"] = check_board(list_ports.comports(), args.port)
            verify_sources(root)
            run_command(base + ["run", "-d", str(project), "-e", ENV, "-t", "upload",
                                "--upload-port", args.port], run_dir, "03_upload", record)
            verify_sources(root)
            for name, value in artifacts.items():
                if digest(project / ".pio/build" / ENV / name) != value:
                    raise PilotError("Бинарный файл изменился во время загрузки: " + name)
            record["board_before_serial"] = check_board(list_ports.comports(), args.port)
            device = serial.Serial(args.port, 115200, timeout=0.5, write_timeout=2)
            capture = SerialCapture(device, run_dir)
            startup = []
            deadline = time.monotonic() + 60
            while True:
                line = capture.line(deadline)
                startup.append(line)
                if line.startswith("ERROR "):
                    raise PilotError(line)
                if line.startswith("READY "):
                    break
            ready = validate_ready(startup)
            record["ready"] = ready
            print("\nПлата готова. В UsbMeter: 10 sps, Start CUR = 0, Stop CUR = 0; запись VBUS, IBUS и PBUS.")
            print("Создайте НОВУЮ запись кнопкой с листом (Create), затем нажмите треугольник Start.")
            input("Когда запись пошла, нажмите Enter здесь. До этого вычисления не начнутся: ")
            record["run_command_utc"] = utc()
            capture.send_run()
            print("Опыт идёт около 3 минут. Не нажимайте Reset, не отключайте кабели и не меняйте настройки FNB58.")
            result_lines = []
            deadline = time.monotonic() + 600
            while True:
                line = capture.line(deadline)
                result_lines.append(line)
                if line.startswith(("ERROR ", "HELLO ", "EXECUTION ")):
                    raise PilotError("Ошибка или перезапуск во время опыта: " + line)
                if line.startswith("WAIT_DONE "):
                    break
            record["result"] = validate_done(result_lines, ready)
            record["status"] = "software_protocol_pass_external_trace_pending"
            code = 0
            print("\nПротокол прошивки прошёл проверку. В UsbMeter нажмите Stop (квадрат), затем Сохранить.")
            print("Имя файла: mega_coeff_pilot_01.cfn. Энергию ещё нужно проверить по этому файлу.")
    except KeyboardInterrupt:
        record["error"] = "interrupted_by_user"
        print("\nПрервано. Измерение не принято. Плата может ещё выполнять начатую серию.", file=sys.stderr)
    except Exception as exc:
        record["error"] = type(exc).__name__ + ": " + str(exc)
        print("\nОСТАНОВКА: " + str(exc), file=sys.stderr)
        print("Дальнейшая загрузка/измерение не выполняются. Логи сохранены.", file=sys.stderr)
    finally:
        if capture:
            capture.close()
        if device:
            device.close()
        record["finished_utc"] = utc()
        write_json(run_dir / "RUN_RECORD.json", record)
        archive = run_dir.with_suffix(".zip")
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(run_dir.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(run_dir.parent).as_posix())
        print("\nАрхив логов: " + str(archive))
        print("Пришлите этот ZIP; после опыта также приложите mega_coeff_pilot_01.cfn.")
    return code


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    raise SystemExit(main())
