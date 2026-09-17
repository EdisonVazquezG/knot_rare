#!/usr/bin/env python3
"""Stage 46: exact diagram-based verification of the Stage 45 width examples.

Python core: Spherogram diagrams, Khoca over QQ, Regina HOMFLY/Jones.
Optional Sage phase: authors' Theta implementation, including actual mirrors.
No archived polynomial or homology enters either computation worker.
See LEEME_stage46.md for scope, installation and interpretation.
"""
from __future__ import annotations
import argparse
import base64
import zlib
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile

VERSION = '46.1.3'
TARGETS = ('14n00036', '14n11437', '14n08809', '15n113508')
PAIRS = (TARGETS[:2], TARGETS[2:])
CONTROLS = ('3_1', '4_1', '5_2')
PACKAGES = ('spherogram==2.4.1', 'snappy_15_knots==1.2.1',
            'khoca==1.5', 'py==1.11.0', 'regina==7.4.1')
THETA_URL = 'https://www.rolandvdv.nl/Theta/Theta.sage'
KNOTS_URL = 'https://www.rolandvdv.nl/Theta/knots.sage'
HERE = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()

# Frozen Stage45 reference: comparison only, never used by computation workers.
REFERENCE_SHA256 = 'ba11415704401929da411f1d48459d6da6a81a81d3c0c05230da3d52019e1d98'
REFERENCE_ZLIB_B64 = (
    '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'
    '6V0z7q93/wPaad9a'
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def clean(d):
    return {k: int(v) for k, v in sorted(d.items()) if v}


def compact_name(name):
    return re.sub(r'n0*(\d+)$', lambda m: 'n' + str(int(m[1])), name)


def reflect_free(d):
    out = {}
    for key, value in d.items():
        q, t = map(int, re.fullmatch(r'F_q(-?\d+)_t(-?\d+)', key).groups())
        out[f'F_q{-q}_t{-t}'] = value
    return clean(out)


def free_stats(d):
    delta, euler = set(), Counter()
    for key, value in d.items():
        q, t = map(int, re.fullmatch(r'F_q(-?\d+)_t(-?\d+)', key).groups())
        if value <= 0:
            raise ValueError('Non-positive free rank')
        delta.add(q - 2*t)
        euler[q] += value * (1 if t % 2 == 0 else -1)
    return dict(rank=sum(d.values()), width=len(delta), diagonals=sorted(delta),
                euler_q={str(k): v for k, v in sorted(euler.items()) if v})


def parse_regina_homfly(poly):
    """Parse exact integer terms and reconstruct in Regina to guard parsing."""
    import regina
    text = str(poly)
    # Splitting on spaced signs preserves negative powers such as x^-4.
    terms = re.split(r' (?=[+-] )', text)
    out, rebuilt = {}, regina.Laurent2()
    for term in terms:
        term = term.strip()
        sign = -1 if term.startswith('-') else 1
        term = term.lstrip('+-').strip()
        tokens = term.split()
        coefficient, a, z = sign, 0, 0
        for token in tokens:
            if re.fullmatch(r'\d+', token):
                coefficient *= int(token)
            else:
                m = re.fullmatch(r'([xy])(?:\^(-?\d+))?', token)
                if not m:
                    raise ValueError(f'Unsupported Regina term: {token!r}')
                power = int(m[2]) if m[2] else 1
                if m[1] == 'x':
                    a = power
                else:
                    z = power
        if coefficient:
            out[f'a{a}_z{z}'] = coefficient
            rebuilt[a, z] = coefficient
    if rebuilt != poly:
        raise ValueError('HOMFLY serialization did not reconstruct the polynomial')
    return clean(out)


def alexander_from_homfly(h):
    """Delta(t) = P(1, t^(1/2)-t^(-1/2)), symmetrically normalized."""
    twice = Counter()
    for key, value in h.items():
        a, z = map(int, re.fullmatch(r'a(-?\d+)_z(-?\d+)', key).groups())
        if z < 0:
            raise ValueError('Negative Conway exponent for a knot')
        for k in range(z + 1):
            twice[z - 2*k] += value * math.comb(z, k) * (-1)**k
    twice = {k: v for k, v in twice.items() if v}
    if any(k % 2 for k in twice):
        raise ValueError('Half-integral Alexander exponent for a knot')
    return normalize_alexander({k//2: v for k, v in twice.items()})


def normalize_alexander(d):
    if not d:
        raise ValueError('Zero Alexander polynomial for a knot')
    span_center = min(d) + max(d)
    if span_center % 2:
        raise ValueError('Non-integral Alexander centering')
    value_at_one = sum(d.values())
    if value_at_one not in (-1, 1):
        raise ValueError('Alexander polynomial does not evaluate to +/-1')
    out = {k-span_center//2: v*value_at_one for k, v in d.items() if v}
    if out != {-k: v for k, v in out.items()}:
        raise ValueError('Alexander symmetry failed')
    return {f'A{k}': v for k, v in sorted(out.items())}


def braid_to_long(word):
    """Closed downward braid -> long-knot crossings and clockwise closures.

    Each visit is an incoming arc. Positive generator: left strand over.
    Each uncut closure contributes -1 rotation; the final outer closure is cut.
    Controls compare this representation against the authors' planar diagrams.
    """
    word = list(map(int, word))
    if not word or 0 in word:
        raise ValueError('Expected a nonempty Artin word')
    n = max(map(abs, word)) + 1
    crossings = [[1 if x > 0 else -1, None, None] for x in word]
    phi = [0] * (2*len(word) + 1)
    visited, current, encounter = set(), 0, 0
    while current not in visited:
        visited.add(current)
        position = current
        for j, generator in enumerate(word):
            left = abs(generator)-1
            if position not in (left, left+1):
                continue
            over = (position == left) == (generator > 0)
            slot = 1 if over else 2
            if crossings[j][slot] is not None:
                raise ValueError('Repeated braid crossing visit')
            crossings[j][slot] = encounter
            encounter += 1
            position = left+1 if position == left else left
        current = position
        if current != 0:
            phi[encounter] -= 1
    if current != 0 or len(visited) != n or encounter != 2*len(word):
        raise ValueError('Braid closure is not a single knot component')
    if any(None in row for row in crossings):
        raise ValueError('Unvisited crossing')
    return [crossings, phi]


def long_to_pd(diagram):
    import regina
    crossings, phi = diagram
    visits = [None] * (2*len(crossings))
    for index, (sign, over, under) in enumerate(crossings, 1):
        if visits[over] is not None or visits[under] is not None:
            raise ValueError('Invalid long-knot crossing visits')
        visits[over], visits[under] = index, -index
    if None in visits or len(phi) != len(visits)+1:
        raise ValueError('Incomplete long-knot diagram')
    knot = regina.Link.fromData([c[0] for c in crossings], [visits])
    return [[int(v) for v in row] for row in knot.pdData()]


def oriented_classical(result, orientation):
    if orientation == 'direct':
        return {v: result[v] for v in ('homfly_regina_alpha_z', 'jones', 'alexander')}
    h = {}
    for k,v in result['homfly_regina_alpha_z'].items():
        a,z = map(int, re.fullmatch(r'a(-?\d+)_z(-?\d+)', k).groups())
        h[f'a{-a}_z{z}'] = v*(-1)**z
    return dict(homfly_regina_alpha_z=h,
                jones={f'J{-int(k[1:])}': v for k,v in result['jones'].items()},
                alexander=result['alexander'])


def mirror_long(diagram):
    crossings, phi = diagram
    return [[[-s, under, over] for s, over, under in crossings], list(phi)]


def author_diagram(text, key):
    pattern = (r'[\"\']' + re.escape(key) +
               r'[\"\']\s*:\s*\[matrix\((\[.*?\])\),\s*(\[.*?\])\]')
    m = re.search(pattern, text)
    if not m:
        raise ValueError(f'No authors\' diagram for {key}')
    return [ast.literal_eval(m[1]), ast.literal_eval(m[2])]


def make_diagram(name, include_braid=False):
    from spherogram import Link
    knot = Link(compact_name(name))
    if len(knot.link_components) != 1:
        raise ValueError(f'{name}: not a knot')
    pd = [[int(v)+1 for v in c] for c in knot.PD_code()]
    expected_crossings = int(re.match(r'\d+', name)[0])
    if len(pd) != expected_crossings:
        raise ValueError(f'{name}: crossing count does not match its table label')
    result = dict(knot=name, source='Spherogram named-link table',
                  table_name=compact_name(name), pd_1_based=pd,
                  dt_code=[[int(v) for v in c] for c in knot.DT_code()],
                  crossing_signs=[int(c.sign) for c in knot.crossings])
    if include_braid:
        word = list(map(int, knot.braid_word()))
        result.update(braid_word=word, braid_long=braid_to_long(word))
    return result


def worker_core(diagram_path, output):
    """Consumes diagrams ONLY; the worker is not given the reference archive."""
    import regina
    from khoca import InteractiveCalculator
    diagram = read(diagram_path)
    pd = diagram['pd_1_based']
    raw = InteractiveCalculator(coefficient_ring=1)(pd)
    free = Counter()
    for degree, quantum, torsion, rank in raw[1]:  # unreduced, over QQ
        if torsion:
            raise ValueError('Unexpected torsion over QQ')
        # Khoca's homological variable is inverse to Sage/Bar-Natan convention.
        free[f'F_q{int(quantum)}_t{-int(degree)}'] += int(rank)
    free = clean(free)
    stats = free_stats(free)
    knot = regina.Link.fromPD(pd)
    h = parse_regina_homfly(knot.homflyAZ())
    j = knot.jones()  # doubled Jones exponents (sqrt(t))
    j2 = {k: int(str(j[k])) for k in range(j.minExp(), j.maxExp()+1)
          if j[k] != 0}
    if any(k % 2 for k in j2):
        raise ValueError('Unexpected odd doubled Jones exponent for a knot')
    expected_euler = Counter()
    for k, v in j2.items():
        expected_euler[k-1] += v
        expected_euler[k+1] += v
    expected_euler = {str(k): v for k, v in sorted(expected_euler.items()) if v}
    if stats['euler_q'] != expected_euler:
        raise ValueError('Khoca Euler characteristic disagrees with Regina Jones')
    save(output, dict(completed=True, knot=diagram['knot'], free=free,
                     free_stats=stats, raw_khoca_unreduced=raw[1],
                     homfly_regina_alpha_z=h,
                     jones={f'J{k//2}': v for k, v in j2.items()},
                     alexander=alexander_from_homfly(h),
                     alexander_method='exact HOMFLY specialization',
                     euler_jones_check=True))


def worker_theta(diagram_path, theta_source, output):
    """Runs within Sage via -c/runpy; reference coefficients are not used."""
    from sage.all import QQ, ZZ, LaurentPolynomialRing, matrix
    from sage.repl.preparse import preparse
    namespace = {}
    exec("from sage.all import *", namespace)
    # Preparse the authors' .sage syntax, including exact rational 1/2.
    exec(preparse(Path(theta_source).read_text()), namespace)
    theta_function = namespace['Theta']
    ring = LaurentPolynomialRing(QQ, names=('t1', 't2'))
    diagram = read(diagram_path)
    values = {}
    for orientation, planar in [('direct', diagram['long']),
                                 ('mirror', mirror_long(diagram['long']))]:
        crossings, phi = planar
        value = ring(theta_function([matrix(ZZ, crossings), list(map(ZZ, phi))]))
        coeff = {}
        for exponents, c in value.dict().items():
            if c.denominator() != 1:
                raise ValueError('Theta has nonintegral coefficients')
            i, j = map(int, exponents)
            coeff[f'T1{i}_T2{j}'] = int(c)
        values[orientation] = clean(coeff)
    save(output, dict(completed=True, knot=diagram['knot'], theta=values,
                     representation=diagram['representation']))


def sage_worker_command(sage, script, diagram_path, theta_source, output):
    """Use Sage's supported -c interface; pass worker options inside sys.argv.

    runpy executes the Python file without Sage-preparsing the entire script.
    Only the downloaded .sage formula is preparsed by worker_theta itself.
    Argument literals preserve spaces and quotes; subprocess uses no shell.
    """
    script = str(Path(script).resolve())
    worker_argv = [script, '--worker-theta', str(diagram_path),
                   str(theta_source), str(output)]
    code = ('import sys, runpy; sys.argv = ' + repr(worker_argv)
            + '; runpy.run_path(' + repr(script) + ", run_name='__main__')")
    return [str(sage), '-c', code]


def run_process(command, log_path, timeout):
    shown = command[:2] if len(command) > 1 and command[1] == '-c' else command[:3]
    print('Running:', ' '.join(map(str, shown)), flush=True)
    with open(log_path, 'w') as log:
        env = os.environ.copy()
        if Path(str(command[0])).name == 'sage':
            env.pop('PYTHONPATH', None)
            env.pop('PYTHONHOME', None)
        process = subprocess.Popen(list(map(str, command)), env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        start = last = time.monotonic()
        while process.poll() is None:
            time.sleep(1)
            elapsed = time.monotonic()-start
            if elapsed > timeout:
                import signal
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise TimeoutError(f'Timeout after {timeout}s; see {log_path}')
            if time.monotonic()-last > 30:
                print(f'  Computing ({elapsed/60:.1f} min); log: {log_path.name}', flush=True)
                last = time.monotonic()
        if process.returncode:
            tail = Path(log_path).read_text(errors='replace')[-2500:]
            raise RuntimeError(f'Worker failed ({process.returncode}): {tail}')


def checkpoint(command, output, fingerprint, timeout):
    output = Path(output)
    metadata = output.with_suffix('.checkpoint.json')
    if output.exists() and metadata.exists():
        meta = read(metadata)
        if (meta.get('fingerprint') == fingerprint and
                meta.get('result_sha256') == sha(output) and
                read(output).get('completed')):
            print(f'Reuse: {output.name}', flush=True)
            return read(output)
    # A failed rerun must never leave an old success consumable downstream.
    metadata.unlink(missing_ok=True)
    output.unlink(missing_ok=True)
    run_process(command, output.with_suffix('.log'), timeout)
    result = read(output)
    if not result.get('completed'):
        raise ValueError('Worker did not finish')
    save(metadata, dict(fingerprint=fingerprint, result_sha256=sha(output)))
    return result


def download(url, path):
    path = Path(path)
    if not path.exists():
        print(f'Download: {path.name}', flush=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        tmp = path.with_suffix(path.suffix+'.tmp')
        tmp.write_bytes(data)
        tmp.replace(path)
    return path


def find_sage(explicit):
    options = [explicit, os.environ.get('SAGE_BIN'), shutil.which('sage'),
               str(Path(os.environ.get('SAGE_ENV', '/content/stage31a_sage_env'))/'bin/sage'),
               '/content/stage46_sage_env/bin/sage']
    for value in options:
        if value and Path(value).is_file() and os.access(value, os.X_OK):
            return str(Path(value).resolve())
    return None


def install_sage(out):
    base = Path('/content') if Path('/content').is_dir() else Path.home()/'.cache'
    prefix = base/'stage46_sage_env'
    binary = prefix/'bin/sage'
    if binary.exists():
        return str(binary)
    bootstrap = base/'stage46_micromamba'
    bootstrap.mkdir(parents=True, exist_ok=True)
    archive = download('https://micro.mamba.pm/api/micromamba/linux-64/latest',
                       bootstrap/'micromamba.tar.bz2')
    mamba = bootstrap/'micromamba'
    with tarfile.open(archive) as tar:
        source = tar.extractfile('bin/micromamba')
        if source is None:
            raise ValueError('Missing micromamba binary')
        mamba.write_bytes(source.read())
    mamba.chmod(0o755)
    os.environ['MAMBA_ROOT_PREFIX'] = str(bootstrap/'root')
    print('Installing Sage in a separate environment; this can require several GB.', flush=True)
    run_process([mamba, 'create', '-y', '-p', prefix, '-c', 'conda-forge',
                 '--strict-channel-priority', 'sage'], out/'install_sage.log', 5400)
    if not binary.exists():
        raise RuntimeError('Sage installation did not create the executable')
    return str(binary)


def load_reference(path=None):
    """Read an explicit source; otherwise use adjacent JSON or embedded reference.

    An explicit missing path remains an error: never silently change user input.
    """
    if path is None:
        adjacent = HERE/'stage46_reference_examples.json'
        path = adjacent if adjacent.is_file() else None
    else:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f'Explicit --stage45 path does not exist: {path}')
    if path is None:
        raw = zlib.decompress(base64.b64decode(REFERENCE_ZLIB_B64, validate=True))
        if hashlib.sha256(raw).hexdigest() != REFERENCE_SHA256:
            raise ValueError('Embedded Stage45 reference failed SHA-256 verification')
        payload = json.loads(raw)
        examples = payload['examples']
        provenance = dict(payload['provenance'])
        provenance['embedded_stage46_reference'] = REFERENCE_SHA256
        print('Reference: embedded Stage45 examples (SHA-256 verified)', flush=True)
    elif path.is_dir():
        # These are known paths: no recursive filename selection across runs.
        examples = [read(path/'examples'/f'example_{i:02d}.json') for i in (1, 2)]
        provenance = {str(path/'examples'/f'example_{i:02d}.json'):
                      sha(path/'examples'/f'example_{i:02d}.json') for i in (1, 2)}
    elif path.suffix == '.zip':
        with zipfile.ZipFile(path) as z:
            examples = []
            for i in (1, 2):
                matches = [n for n in z.namelist() if n.endswith(f'/example_{i:02d}.json')
                           or n == f'example_{i:02d}.json']
                if len(matches) != 1:
                    raise ValueError(f'Ambiguous/missing example {i} in {path}')
                examples.append(json.loads(z.read(matches[0])))
        provenance = {str(path): sha(path)}
    else:
        payload = read(path)
        examples, provenance = payload['examples'], payload['provenance']
        provenance[str(path)] = sha(path)
    expected = {}
    pairs = []
    for example in examples:
        pairs.append((example['knot_a'], example['knot_b']))
        for side in ('a', 'b'):
            name = example['knot_'+side]
            expected[name] = dict(
                free=clean({k: v for k, v in example['khovanov']['knot_'+side].items()
                            if k.startswith('F_')}),
                inputs={view: values['knot_'+side] for view, values in example['inputs'].items()})
    if set(expected) != set(TARGETS) or set(pairs) != set(PAIRS):
        raise ValueError('Reference does not contain the frozen Stage45 width pairs')
    return expected, provenance


def compare_core(result, expected):
    f = result['free']; ef = expected['free']
    free_match = 'direct' if f == ef else ('reflected' if reflect_free(f) == ef else 'mismatch')
    h = result['homfly_regina_alpha_z']; eh = expected['inputs']['HOMFLY-PT']
    # Report convention relation explicitly; do not fit shifts or per-term changes.
    hm = {}
    for k, v in h.items():
        a, z = map(int, re.fullmatch(r'a(-?\d+)_z(-?\d+)', k).groups())
        hm[f'a{-a}_z{z}'] = v*((-1)**z)
    h_match = 'direct_alpha' if h == eh else ('alpha_inverted_z_negated' if hm == eh else 'mismatch')
    j = result['jones']; ej = expected['inputs']['Jones']
    jm = {f'J{-int(k[1:])}': v for k, v in j.items()}
    j_match = 'direct' if j == ej else ('inverted' if jm == ej else 'mismatch')
    ea = normalize_alexander({int(k[1:]): v for k, v in expected['inputs']['Alexander'].items()})
    a_match = result['alexander'] == ea
    return dict(free_match=free_match, homfly_match=h_match, jones_match=j_match,
                alexander_match_up_to_units=a_match,
                archival_comparison_pass=(free_match != 'mismatch' and
                    h_match != 'mismatch' and j_match != 'mismatch' and a_match))


def check_small_controls(results):
    # Published rational, unreduced Kh polynomials, up to mirror.
    trefoil = {'F_q-1_t0': 1, 'F_q-3_t0': 1, 'F_q-5_t-2': 1, 'F_q-9_t-3': 1}
    figure8 = {'F_q-5_t-2': 1, 'F_q-1_t-1': 1, 'F_q-1_t0': 1,
               'F_q1_t0': 1, 'F_q1_t1': 1, 'F_q5_t2': 1}
    for name, expected in [('3_1', trefoil), ('4_1', figure8)]:
        f = results[name]['free']
        if f != expected and reflect_free(f) != expected:
            raise ValueError(f'{name}: rational unreduced Kh positive control failed')
    return True


def build_report(core, reference, theta, controls_passed, theta_controls_passed, errors):
    rows = []
    for name in TARGETS:
        row = {'knot': name, 'core_completed': name in core}
        if name in core:
            row.update(core[name]['free_stats'])
            row.update(compare_core(core[name], reference[name]))
        if name in theta:
            expected = reference[name]['inputs']['Theta']
            row['theta_archival_matching_representatives'] = [
                sign for sign, value in theta[name]['theta'].items() if value == expected]
            row['theta_completed'] = True
        else:
            row['theta_completed'] = False
        rows.append(row)
    by_name = {row['knot']: row for row in rows}
    pairs = []
    for a, b in PAIRS:
        pair = dict(knot_a=a, knot_b=b, core_pair_verified=False,
                    exact_four_view_pair_verified=False)
        if a in core and b in core:
            views = ('homfly_regina_alpha_z', 'jones', 'alexander')
            equal = all(core[a][view] == core[b][view] for view in views)
            widths = [core[a]['free_stats']['width'], core[b]['free_stats']['width']]
            pair.update(three_classical_polynomials_equal=equal, widths=widths,
                        width_difference=widths[0] != widths[1],
                        core_pair_verified=bool(controls_passed and equal and widths[0] != widths[1]
                            and all(by_name[n]['archival_comparison_pass'] for n in (a,b))))
            # To align with the archive, use the same recorded classical convention
            # relation for both members; no independent mirror flips of input views.
            same_relations = all(by_name[a][key] == by_name[b][key]
                                 for key in ('homfly_match', 'jones_match'))
            pair['same_classical_convention_relations'] = same_relations
            if a in theta and b in theta and theta_controls_passed:
                ta, tb = theta[a]['theta'], theta[b]['theta']
                both_match = all(by_name[n].get('theta_archival_matching_representatives') for n in (a,b))
                # Exact computed equality, not just agreement with a stored vector.
                orientations = [[oa, ob] for oa, va in ta.items() for ob, vb in tb.items()
                                if va == vb == reference[a]['inputs']['Theta']
                                and vb == reference[b]['inputs']['Theta']
                                and oriented_classical(theta[a]['same_diagram_core'], oa) ==
                                    oriented_classical(theta[b]['same_diagram_core'], ob)]
                pair.update(theta_equal_representatives=orientations,
                    theta_reference_matches=bool(both_match),
                    exact_four_view_pair_verified=bool(pair['core_pair_verified'] and
                        same_relations and both_match and orientations))
        pairs.append(pair)
    core_ok = controls_passed and all(p['core_pair_verified'] for p in pairs)
    full_ok = core_ok and theta_controls_passed and all(p['exact_four_view_pair_verified'] for p in pairs)
    return dict(stage=VERSION, status='FULL_VERIFIED' if full_ok else ('CORE_VERIFIED_THETA_PENDING' if core_ok else 'INCOMPLETE_OR_FAILED'),
                scope='two retrospectively selected exact-fiber width examples',
                control_checks_passed=controls_passed,
                theta_representation_controls_passed=theta_controls_passed,
                knots=rows, pairs=pairs, errors=errors,
                novelty_verified=False, anomaly_detector_discovery_demonstrated=False,
                independent_test_set_validation=False,
                note='FULL_VERIFIED certifies these computations, not a new theorem or priority claim.')


def make_review_zip(out):
    dest = out/'stage46_review.zip'
    with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if (p.is_file() and p != dest and not any(part in ('_deps',) for part in p.relative_to(out).parts)
                    and p.suffix not in ('.bz2', '.gz')):
                z.write(p, p.relative_to(out))
    return dest


def ensure_python_packages(out, argv, force=False):
    """Probe real imports in a fresh process; bootstrap once, then launch cleanly.

    This also works with IPython %run, where sys.modules may contain packages
    from earlier cells. ABI-specific directories prevent mixing Python versions.
    """
    deps = (out/'_deps'/sys.implementation.cache_tag).resolve()
    env = os.environ.copy()
    prior_paths = [part for part in env.get('PYTHONPATH', '').split(os.pathsep) if part]
    env['PYTHONPATH'] = os.pathsep.join([str(deps), *prior_paths])
    token = digest([str(SELF), sha(SELF), sys.executable, str(deps)])
    already_started = env.get('KNOT_STAGE46_READY') == token
    probe_code = (
        'import importlib, importlib.metadata, json\n'
        'versions, errors = {}, {}\n'
        f'for requirement in {PACKAGES!r}:\n'
        '    name, expected = requirement.split("==")\n'
        '    try:\n'
        '        version = importlib.metadata.version(name)\n'
        '        importlib.import_module(name)\n'
        '        versions[name] = version\n'
        '        if version != expected:\n'
        '            errors[name] = f"version {version}, expected {expected}"\n'
        '    except Exception as exc:\n'
        '        errors[name] = f"{type(exc).__name__}: {exc}"\n'
        'print(json.dumps(dict(versions=versions, errors=errors)))\n'
    )
    def probe():
        result = subprocess.run([sys.executable, '-c', probe_code], env=env,
                                capture_output=True, text=True, timeout=120)
        if result.returncode:
            return dict(versions={}, errors={'interpreter': result.stderr[-2500:]})
        try:
            return json.loads(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return dict(versions={}, errors={'probe': result.stdout[-2500:]})
    report = probe()
    if already_started and report['errors']:
        raise RuntimeError(f'Dependencies failed after clean launch: {report["errors"]}')
    if not already_started and (force or report['errors']):
        print('Preparing dependencies automatically for:', sys.executable, flush=True)
        for name, error in report['errors'].items():
            print(f'  {name}: {error}', flush=True)
        deps.mkdir(parents=True, exist_ok=True)
        run_process([sys.executable, '-m', 'pip', 'install', '--upgrade',
                     '--disable-pip-version-check', '--target', deps, *PACKAGES],
                    out/'install_python.log', 1800)
        report = probe()
    save(out/'environment_check.json', dict(python=sys.version,
         executable=sys.executable, dependency_directory=str(deps), **report))
    if report['errors']:
        raise RuntimeError('Dependency installation/import check failed. '
                           f'See {out / "environment_check.json"} and '
                           f'{out / "install_python.log"}. Details: {report["errors"]}')
    if not already_started:
        env['KNOT_STAGE46_READY'] = token
        print('Dependencies verified. Launching Stage 46 in a clean Python process.', flush=True)
        result = subprocess.run([sys.executable, '-u', str(SELF), *argv], env=env)
        if result.returncode == 3:
            # Notebook frontends may hide the child process stdout. Surface the
            # exact Theta error before returning the deliberately partial status.
            summary_path = out/'verification_summary.json'
            if summary_path.is_file():
                summary = read(summary_path)
                print('Stage 46 status:', summary.get('status'), flush=True)
                print(json.dumps(summary.get('errors', []), indent=2, ensure_ascii=False), flush=True)
                print('Review ZIP:', out/'stage46_review.zip', flush=True)
        raise SystemExit(result.returncode)
    return report['versions']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage45', type=Path, help='Explicit Stage45 output directory, review zip, or reference JSON')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--install-deps', action='store_true', help='Force dependency reinstall; missing or incompatible packages install automatically')
    parser.add_argument('--theta', choices=('auto', 'required', 'skip'), default='auto')
    parser.add_argument('--sage', help='Path to Sage executable')
    parser.add_argument('--install-sage', action='store_true', help='Install Sage in a separate environment if absent')
    parser.add_argument('--timeout', type=int, default=7200, help='Seconds per calculation process')
    parser.add_argument('--force', action='store_true', help='Invalidate successful calculation checkpoints')
    args = parser.parse_args(argv)
    root = args.output_root
    if root is None:
        data = Path(os.environ.get('KNOT_DATA_DIR', '/content/drive/MyDrive/Colab Notebooks/data_invariants/Invariants'))
        root = Path(os.environ.get('KNOT_OUTPUT_DIR', str(data/'processed_consensus_hardness/corrected_run_20260819')))
    out = root/'46_independent_diagram_verification'
    for sub in ('diagrams', 'calculations', 'sources'):
        (out/sub).mkdir(parents=True, exist_ok=True)
    print(f'Stage {VERSION}; output: {out}', flush=True)
    reference, ref_provenance = load_reference(args.stage45)
    packages = ensure_python_packages(out, list(sys.argv[1:] if argv is None else argv),
                                      force=args.install_deps)
    save(out/'manifest.json', dict(stage=VERSION, utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version, packages=packages, reference=ref_provenance,
        script_sha256=sha(SELF), argv=list(sys.argv),
        worker_inputs='diagrams only; no archived invariants',
        grading='Khoca homological degree negated; q retained; QQ, unreduced',
        theta_urls=[THETA_URL, KNOTS_URL]))
    shutil.copyfile(SELF, out/SELF.name)
    save(out/'reference_for_comparison_only.json', reference)
    diagrams, core, theta, errors = {}, {}, {}, []
    sage = find_sage(args.sage)
    if not sage and args.install_sage:
        sage = install_sage(out)
    do_theta = args.theta != 'skip' and sage is not None
    if args.theta == 'required' and not sage:
        print('Sage missing: core will run, but full verification remains pending.', flush=True)
    for name in (*CONTROLS, *TARGETS):
        try:
            diagram = make_diagram(name, include_braid=do_theta)
            diagrams[name] = diagram
            dp = out/'diagrams'/f'{name}.json'
            save(dp, diagram)
            result_path = out/'calculations'/f'{name}_core.json'
            fingerprint = digest([sha(SELF), packages, diagram, 'core'])
            if args.force:
                result_path.with_suffix('.checkpoint.json').unlink(missing_ok=True)
            result = checkpoint([sys.executable, SELF, '--worker-core', dp, result_path],
                                result_path, fingerprint, args.timeout)
            core[name] = result
            print(f'{name}: rank={result["free_stats"]["rank"]}, W_F={result["free_stats"]["width"]}', flush=True)
        except Exception as exc:
            errors.append(dict(knot=name, phase='core', error=str(exc)))
            print(f'ERROR {name}: {exc}', flush=True)
    controls_passed = False
    try:
        controls_passed = check_small_controls(core)
    except Exception as exc:
        errors.append(dict(phase='controls', error=str(exc)))
    theta_controls_passed = False
    if do_theta and controls_passed:
        try:
            source = download(THETA_URL, out/'sources'/'Theta.sage')
            knot_source = download(KNOTS_URL, out/'sources'/'knots.sage')
            knot_text = knot_source.read_text()
            save(out/'sources'/'provenance.json', dict(theta_url=THETA_URL,
                theta_sha256=sha(source), knots_url=KNOTS_URL, knots_sha256=sha(knot_source),
                sage=sage, sage_version=subprocess.check_output([sage, '--version'], text=True).strip()))
            source_identity = read(out/'sources'/'provenance.json')
            def calculate_theta(name, representation, long_diagram):
                payload = dict(knot=name, representation=representation, long=long_diagram)
                dp = out/'diagrams'/f'{name}_theta_{representation}.json'
                save(dp, payload)
                rp = out/'calculations'/f'{name}_theta_{representation}.json'
                fingerprint = digest([sha(SELF), source_identity, payload, 'theta'])
                if args.force:
                    rp.with_suffix('.checkpoint.json').unlink(missing_ok=True)
                # Recompute the other invariants on THIS Theta diagram too. This
                # prevents combining unrelated mirror representatives across views.
                core_dp = out/'diagrams'/f'{name}_theta_{representation}_core.json'
                core_payload = dict(knot=name, pd_1_based=long_to_pd(long_diagram))
                save(core_dp, core_payload)
                core_rp = out/'calculations'/f'{name}_theta_{representation}_core.json'
                core_fp = digest([sha(SELF), packages, core_payload, 'theta_diagram_core'])
                if args.force:
                    core_rp.with_suffix('.checkpoint.json').unlink(missing_ok=True)
                same_core = checkpoint([sys.executable, SELF, '--worker-core', core_dp, core_rp],
                                       core_rp, core_fp, args.timeout)
                known_free = core[name]['free']
                if same_core['free'] not in (known_free, reflect_free(known_free)):
                    raise ValueError(f'{name}: Theta diagram has different free Kh from named PD')
                value = checkpoint(sage_worker_command(sage, SELF, dp, source, rp),
                                   rp, fingerprint, args.timeout)
                value['same_diagram_core'] = same_core
                return value
            # Check braid rotation encoding against three independent planar diagrams.
            theta_control_results = {}
            for name in CONTROLS:
                direct = calculate_theta(name, 'authors', author_diagram(knot_text, name))
                braid = calculate_theta(name, 'braid', diagrams[name]['braid_long'])
                same_orbit = sorted(digest(v) for v in direct['theta'].values()) == sorted(digest(v) for v in braid['theta'].values())
                if not same_orbit:
                    raise ValueError(f'{name}: braid/rotation Theta control failed')
                theta_control_results[name] = dict(same_mirror_orbit=True)
            theta_controls_passed = True
            save(out/'theta_controls.json', theta_control_results)
            for name in TARGETS:
                try:
                    # Author diagrams through 14; independently constructed braid at 15.
                    if name.startswith('14'):
                        rep, long = 'authors', author_diagram(knot_text, compact_name(name))
                    else:
                        rep, long = 'braid', diagrams[name]['braid_long']
                    theta[name] = calculate_theta(name, rep, long)
                except Exception as exc:
                    errors.append(dict(knot=name, phase='theta', error=str(exc)))
        except Exception as exc:
            errors.append(dict(phase='theta_setup_or_controls', error=str(exc)))
    else:
        errors.append(dict(phase='theta', error='Skipped or Sage unavailable; no independent Theta certificate.'))
    report = build_report(core, reference, theta, controls_passed, theta_controls_passed, errors)
    save(out/'verification_summary.json', report)
    import csv
    fields = ['knot', 'core_completed', 'rank', 'width', 'diagonals', 'free_match',
              'homfly_match', 'jones_match', 'alexander_match_up_to_units',
              'archival_comparison_pass', 'theta_completed', 'theta_archival_matching_representatives']
    with (out/'knot_verification.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(report['knots'])
    if all(name in core for name in TARGETS):
        differences = []
        for a,b in PAIRS:
            # Use the archive's documented grading representative only after computation.
            vectors = []
            for name in (a,b):
                f = core[name]['free']
                if compare_core(core[name], reference[name])['free_match'] == 'reflected':
                    f = reflect_free(f)
                vectors.append(f)
            delta = clean({k: vectors[0].get(k,0)-vectors[1].get(k,0)
                           for k in set(vectors[0])|set(vectors[1])})
            differences.append(dict(knot_a=a,knot_b=b,free_poincare_difference=delta))
        save(out/'independent_poincare_differences.json', differences)
    review = make_review_zip(out)
    print('\nSTATUS:', report['status'], '\nReview ZIP:', review, flush=True)
    print('Novelty/priority is not evaluated. See verification_summary.json for the exact scope.')
    if report['status'] == 'INCOMPLETE_OR_FAILED':
        raise SystemExit(2)
    if args.theta == 'required' and report['status'] != 'FULL_VERIFIED':
        raise SystemExit(3)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker-core':
        worker_core(*sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == '--worker-theta':
        worker_theta(*sys.argv[2:])
    else:
        main()
