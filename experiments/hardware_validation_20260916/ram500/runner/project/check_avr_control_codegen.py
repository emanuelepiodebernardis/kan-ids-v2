"""Check actual AVR heap CONTROL code, not HOST_CHECK substitutes.

Usage: python check_avr_control_codegen.py --root PROJECT/validation \
    --source PROJECT/src/main.cpp --out codegen_acceptance.json

Finds firmware.disassembly.txt below --root and requires all five model variants.
The binary checks reject the received v0.14.0 defect. Predicate checks refer to
the supplied corresponding source; this is bounded regression checking, not
formal verification of arbitrary compiler output or a hardware run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

MODELS = {'coeff', 'lut', 'mlp', 'kanml', 'dt5'}

def instructions(text):
    result = []
    for line in text.splitlines():
        match = re.match(r'^\s*([0-9a-f]+):\s+((?:[0-9a-f]{2}\s+)+)\s*(\S+)\s*(.*)', line)
        if match:
            result.append({'address': int(match[1], 16), 'op': match[3], 'tail': match[4], 'line': line.strip()})
    return result

def named(insn, symbol):
    return bool(re.search(r'<' + re.escape(symbol) + r'(?:\+0x[0-9a-f]+)?>', insn['tail']))

def indices(code, symbol, op=None, lo=0, hi=None):
    return [i for i in range(lo, len(code) if hi is None else hi)
            if named(code[i], symbol) and (op is None or code[i]['op'] == op)]

def model_of(path):
    for part in reversed(path.parts):
        if part in MODELS:
            return part
        for model in MODELS:
            if part == 'mega_ram_' + model:
                return model
    return None

def check_text(text):
    code = instructions(text)
    failures = []
    evidence = {}
    try:
        before = indices(code, 'ram_control_heap_before', 'sts')[0]
        during = indices(code, 'ram_control_heap_during', 'sts', before+1)[0]
        after = indices(code, 'ram_control_heap_after', 'sts', during+1)[0]
        ok = indices(code, 'ram_control_heap_ok', 'sts', after+1)[0]
        malloc = indices(code, '__wrap_malloc', 'call', before+1, during)[0]
        free = indices(code, '__wrap_free', 'call', during+1, after)[0]
        # A direct LDS fetch of each byte of the current heap pointer is
        # required in every snapshot interval. No saved-register substitution.
        snapshots = {
            'before_malloc': indices(code, '__brkval', 'lds', max(0, before-32), before),
            'after_malloc_before_during_store': indices(code, '__brkval', 'lds', malloc+1, during),
            'after_free_before_after_store': indices(code, '__brkval', 'lds', free+1, after),
        }
        evidence['snapshot_loads'] = {k: [code[i]['line'] for i in v] for k,v in snapshots.items()}
        for name, reads in snapshots.items():
            if len(reads) < 2 or not any('<__brkval>' in code[i]['tail'] for i in reads) or not any('<__brkval+0x1>' in code[i]['tail'] for i in reads):
                failures.append(name + ': missing a fresh full 16-bit pointer read')
        evidence['allocator_calls'] = [code[malloc]['line'], code[free]['line']]
        # The failing build folds the result to STS ...,r1. Search only until
        # the first subsequent call (start of printing), not unrelated code.
        stop = next((i for i in range(ok+1, len(code)) if code[i]['op'] in ('call', 'rcall')), min(ok+20,len(code)))
        stores = indices(code, 'ram_control_heap_ok', 'sts', after+1, stop)
        evidence['heap_ok_stores'] = [code[i]['line'] for i in stores]
        if not any(not re.search(r',\s*r1\s*(?:;|$)', code[i]['tail']) for i in stores):
            failures.append('heap_ok is stored only as constant zero (AVR zero register r1)')
        for symbol, start in [('ram_malloc_calls', malloc+1), ('ram_free_calls', free+1)]:
            reads = indices(code, symbol, 'lds', start, ok)
            evidence[symbol + '_reads_after_operation'] = [code[i]['line'] for i in reads]
            if len(reads) < 2:
                failures.append(symbol + ': missing full post-operation counter observation')
    except IndexError:
        failures.append('required CONTROL symbols/calls/stores not found in expected order')
    evidence['failures'] = failures
    evidence['passed'] = not failures
    return evidence

def audit(listing, nm=None):
    """Build-hook entry point; input is actual avr-objdump disassembly text.

    Returns evidence on success, raises ValueError on any failed codegen gate.
    nm is accepted for hook compatibility but symbol-tagged objdump is enough.
    """
    result = check_text(listing)
    if not result['passed']:
        raise ValueError('AVR heap CONTROL codegen failed: ' + '; '.join(result['failures']))
    return result

def check_file(path):
    result = check_text(path.read_text(encoding='utf-8'))
    result.update(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return result

def source_checks(path):
    code = path.read_text(encoding='utf-8')
    code = re.sub(r'/\*.*?\*/|//[^\n]*', '', code, flags=re.S)
    patterns = {
        'heap_growth_128': r'ram_control_heap_during\s*>=\s*ram_control_heap_before\s*\+\s*128\b',
        'malloc_counter_increment_one': r'(?:ram_malloc_calls\s*==\s*mc\s*\+\s*1\b|heap_malloc_calls\s*==\s*1\b)',
        'free_counter_increment_one': r'(?:ram_free_calls\s*==\s*fc\s*\+\s*1\b|heap_free_calls\s*==\s*1\b)',
        'nonnull_probe': r'(?:ram_control_heap_ok|heap_alloc_ok)\s*=\s*\(?\s*p\s*!=\s*nullptr',
        'allocation_size_128': r'malloc\s*\(\s*128\s*\)',
    }
    checks = {key: bool(re.search(pattern, code)) for key,pattern in patterns.items()}
    if re.search(r'heap_malloc_calls\s*==\s*1\b', code):
        checks['malloc_delta_measured'] = bool(re.search(r'heap_malloc_calls\s*=\s*\(uint16_t\)\s*\(\s*ram_malloc_calls\s*-\s*mc\s*\)', code))
        checks['free_delta_measured'] = bool(re.search(r'heap_free_calls\s*=\s*\(uint16_t\)\s*\(\s*ram_free_calls\s*-\s*fc\s*\)', code))
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'checks': checks, 'passed': all(checks.values())}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    checked = []
    found = set()
    for path in sorted(args.root.rglob('firmware.disassembly.txt')):
        model = model_of(path)
        if model is None:
            continue
        result = check_file(path)
        result['model'] = model
        found.add(model)
        checked.append(result)
    source = source_checks(args.source)
    result = {'schema': 'kanids-avr-heap-control-codegen-v1', 'model_set_complete': found == MODELS,
              'files_checked': len(checked), 'source_predicates': source, 'binaries': checked,
              'passed': found == MODELS and source['passed'] and all(item['passed'] for item in checked),
              'scope': 'Bounded code-generation regression and source predicate preservation; on-device CONTROL remains mandatory.'}
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({'passed': result['passed'], 'models': sorted(found), 'files_checked': len(checked),
                      'failures': {str(Path(item['path']).relative_to(args.root)): item['failures'] for item in checked if not item['passed']},
                      'source_checks': source['checks']}, indent=2))
    return 0 if result['passed'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
