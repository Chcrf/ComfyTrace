"""Behavior regressions for carving findings F1-F7, F9, O1 and contained minors."""

import hashlib
import io
import json
import struct
import time
import tracemalloc
import zlib
from functools import lru_cache
from types import SimpleNamespace

import pytest
from PIL import Image

from comfytrace.carve import (
    MAX_JSON_CANDIDATE_BYTES,
    PNG_MAGIC,
    Ihdr,
    PhysicalSource,
    PngCarver,
    ProcessLayerSource,
    SpanSet,
    StreamReader,
    WorkflowCarver,
    canonical_json,
)
from comfytrace.cli import _carve, _export
from comfytrace.correlate import KeyMatcher, correlate
from comfytrace.ingest import SymbolManager
from comfytrace.report import EvidentiaryExporter
from tests.synthetic import PAGE, FakeProcLayer, _chunk, api_prompt, noise_image, png_bytes, prompt_request_body


def _mapped_png(png, missing):
    count = (len(png) + PAGE - 1) // PAGE
    layer = FakeProcLayer(png.ljust(count * PAGE, b'\x01'), {})
    base = 0x20000000
    layer.map_virtual(base, list(range(count)))
    for page in missing:
        del layer.page_table[base // PAGE + page]
    return layer, base


@pytest.mark.parametrize('missing', [(2,), (2, 3)])
@pytest.mark.parametrize('chunk_size', [700, 50000])
def test_f1_process_holes_keep_exact_gaps_and_later_spans(tmp_path, missing, chunk_size):
    png = png_bytes(noise_image(100, 100), {'prompt_id': 'before'}, idat_chunk=chunk_size)
    png = png[:-12] + _chunk(b'tEXt', b'client_id\x00after') + png[-12:]
    layer, base = _mapped_png(png, missing)
    [record] = PngCarver().carve_stream(
        png[:PAGE], virtual_address=base, process_id=42, source=ProcessLayerSource(layer)
    )
    start, end = missing[0] * PAGE, (missing[-1] + 1) * PAGE
    # A hole inside one chunk's body keeps the chain; a hole that swallows a chunk header ends the image,
    # because bytes after it can no longer be tied to this PNG.
    chained = chunk_size == 50000
    pieces = [png[:start], png[end:]] if chained else [png[:start]]
    assert record.completeness == 'PARTIAL'
    assert [(g.relative_offset, g.length_bytes, str(g.reason)) for g in record.gaps] == [
        (start, end - start, 'PAGE_NOT_PRESENT')
    ]
    assert record.spans == list(zip([0, end][:len(pieces)], pieces, strict=True))
    assert record.recovered_bytes == sum(map(len, pieces))
    assert record.embedded_text == ({'prompt_id': 'before', 'client_id': 'after'} if chained
                                    else {'prompt_id': 'before'})
    assert record.validation == {'crc32_valid': False, 'scanlines_valid': False, 'has_gaps': True}
    assert record.derived_preview_metadata['scanline_halt_reason'] == 'PAGE_GAP_ENCOUNTERED'
    _export(tmp_path, [record], [])
    assert [path.rsplit('/', 1)[-1] for path in record.raw_file_paths] == [
        f'span_{i:03d}.bin' for i in range(len(pieces))]
    assert [(tmp_path / p).read_bytes() for p in record.raw_file_paths] == pieces
    assert record.derived_preview_metadata['raw_span_sha256'] == [
        hashlib.sha256(piece).hexdigest() for _, piece in record.spans
    ]
    paths = EvidentiaryExporter(tmp_path / 'direct').export_raw_artifact(record, record.spans[0][1])
    assert [(tmp_path / 'direct' / p).read_bytes() for p in paths] == pieces


def test_f1_physical_tears_still_stop_without_later_spans():
    png = bytearray(png_bytes(noise_image(100, 100), idat_chunk=700))
    png[2 * PAGE:3 * PAGE] = b'\xff' * PAGE
    [record] = PngCarver(physical_mode=True).carve_stream(bytes(png), source=PhysicalSource(bytes(png)))
    assert record.gaps[0].reason == 'NONCONTIGUOUS_PHYSICAL'
    assert len(record.spans) == 1
    assert record.spans[0][1] == png[:record.recovered_bytes]
    assert record.recovered_bytes <= 2 * PAGE


def test_f1_hole_splitting_a_chunk_header_stops_at_the_hole():
    png = png_bytes(noise_image(100, 100), idat_chunk=700)
    prefix_size = 2 * PAGE - 35
    memory = b'\x01' * prefix_size + png
    layer, base = _mapped_png(memory, (2,))
    [record] = PngCarver().carve_stream(
        memory[:2 * PAGE], virtual_address=base, process_id=42, source=ProcessLayerSource(layer)
    )
    relative = 2 * PAGE - prefix_size
    assert record.spans == [(0, png[:relative])]
    assert [(g.relative_offset, g.length_bytes) for g in record.gaps] == [(relative, PAGE)]


class _CachedLayer(FakeProcLayer):
    @lru_cache(maxsize=512)  # noqa: B019 - reproduces Volatility's retained read buffers
    def read(self, offset, length, pad=False):
        return super().read(offset, length, pad=pad)


def test_f2_process_reads_do_not_retain_block_buffers(tmp_path):
    layer = _CachedLayer(b'x' * (32 * PAGE), {})
    base = 0x100000
    layer.map_virtual(base, list(range(32)))
    source = ProcessLayerSource(layer)
    try:
        assert source.read_prefix(base, 8 * PAGE) == (b'x' * (8 * PAGE), None)
        assert layer.read.cache_info().currsize == 0
        assert source.read_before(base + 8 * PAGE, 4 * PAGE) == b'x' * (4 * PAGE)
        assert layer.read.cache_info().currsize == 0
        blocks = StreamReader(tmp_path / 'unused').iter_vad_blocks(
            layer, base, base + 32 * PAGE - 1, block_size=8 * PAGE, overlap=PAGE
        )
        for block in blocks:
            assert block.data and not block.gaps
            assert layer.read.cache_info().currsize == 0
        assert all(not pad for _, _, pad in layer.reads)
    finally:
        layer.read.cache_clear()


@pytest.mark.parametrize('runtime', [False, True])
def test_f3_lone_surrogate_candidate_does_not_abort_later_workflow(runtime):
    bad = (prompt_request_body(api_prompt(), '\ud800') if runtime
           else json.dumps(api_prompt(text='\ud800')).encode())
    good = json.dumps(api_prompt(seed=19)).encode()
    records = WorkflowCarver().carve_stream(bad + b'\x00' + good)
    assert any(r.raw_bytes == good for r in records)
    assert all(r.raw_bytes != bad for r in records)
    [image] = PngCarver().carve_stream(png_bytes(noise_image(4, 4), {'prompt': bad.decode()}))
    result = correlate([image], records)
    if not runtime:
        assert result.links == []
    assert all(r.execution_parameters.get('prompt_id') != '\ud800' for r in records)


def test_f4_nested_overlapping_candidates_are_fast_and_bounded():
    graph = api_prompt()
    junk = b'{"x":' * 800 + b'"' + b'A' * 1_000_000 + b'"' + b'}' * 800
    blob = b'{"junk":' + junk + b',"prompt":' + json.dumps(graph).encode() + b'}'
    start = time.perf_counter()
    [record] = WorkflowCarver().carve_stream(blob)
    elapsed = time.perf_counter() - start
    assert record.graph == graph and record.raw_bytes == blob
    assert elapsed < 1.0, f'{elapsed:.3f} seconds for {len(blob)} bytes'
    tracemalloc.start()
    try:
        [measured] = WorkflowCarver().carve_stream(blob)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert measured.raw_bytes == blob
    assert peak < 32 * 1024 * 1024, f'{peak} traced bytes'


def test_f4_nested_unrecognized_wrappers_share_candidate_work():
    graph = api_prompt(text='x' * 1_000_000)
    body = json.dumps(graph).encode()
    blob = b'{"payload":' * 800 + body + b'}' * 800
    start = time.perf_counter()
    [record] = WorkflowCarver().carve_stream(blob)
    elapsed = time.perf_counter() - start
    assert record.graph == graph and record.raw_bytes == body
    assert elapsed < 1.0, f'{elapsed:.3f} seconds for {len(blob)} bytes'


@pytest.mark.parametrize('codec,unit', [('utf-8', 1), ('utf-16-le', 2)])
def test_f4_ten_megabyte_candidate_cap_still_preserves_bounded_fragments(codec, unit):
    graph = api_prompt(text='x' * (MAX_JSON_CANDIDATE_BYTES // unit))
    body = json.dumps(graph).encode(codec)
    assert len(body) > MAX_JSON_CANDIDATE_BYTES
    records = WorkflowCarver().carve_stream(body)
    assert records and all(r.completeness == 'PARTIAL' for r in records)
    assert all(len(r.raw_bytes) <= MAX_JSON_CANDIDATE_BYTES and r.raw_bytes != body for r in records)


@pytest.mark.parametrize('suffix', [_chunk(b'IEND', b'nonempty'), _chunk(b'ABCD', b'unknown')])
def test_f5_invalid_critical_chunks_are_never_complete(suffix):
    png = png_bytes(noise_image(4, 4))
    malformed = png[:-12] + suffix + (png[-12:] if suffix[4:8] != b'IEND' else b'')
    [record] = PngCarver().carve_stream(malformed)
    assert record.completeness == 'PARTIAL'
    assert record.validation['scanlines_valid'] is False
    assert record.validation['crc32_valid'] is True
    assert all(g.length_bytes > 0 for g in record.gaps)


def test_f6_unverified_chunk_tails_never_claim_crc_validation():
    png = png_bytes(noise_image(20, 20))
    for tail in (struct.pack('>I', 0x7fffffff) + b'IDAT' + b'garbage' * 1000, png[33:90]):
        [record] = PngCarver().carve_stream(png[:33] + tail)
        assert record.validation['crc32_valid'] is False
        assert record.spans == [(0, png[:33] + tail)]


def test_f6_pixel_failure_does_not_erase_verified_crcs():
    ihdr = _chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 0, 0, 0, 0))
    compressed = zlib.compress(b'\x00\x7f')
    png = (PNG_MAGIC + ihdr + _chunk(b'IDAT', compressed[:3]) + _chunk(b'tEXt', b'x\x00y')
           + _chunk(b'IDAT', compressed[3:]) + _chunk(b'IEND', b''))
    [record] = PngCarver().carve_stream(png)
    assert record.completeness == 'PARTIAL'
    assert record.validation['crc32_valid'] is True
    assert record.gaps == [] and record.validation['has_gaps'] is False


def test_f7_nested_pngs_do_not_unexclude_outer_metadata():
    inner = png_bytes(noise_image(1, 1))
    outer = png_bytes(noise_image(4, 4), {'prompt': json.dumps(api_prompt())})
    outer = outer[:33] + _chunk(b'raNd', inner) + outer[33:]
    carver = PngCarver()
    assert len(carver.carve_stream(outer)) == 2
    assert WorkflowCarver().carve_stream(outer, exclude=carver.claimed['physical']) == []
    spans = SpanSet([(0, 100), (10, 20), (90, 120), (120, 130)])
    assert all(p in spans for p in range(130))
    assert -1 not in spans and 130 not in spans


def test_f9_identical_workflows_keep_all_process_copy_locations(tmp_path):
    body = prompt_request_body(api_prompt())
    spaces = []
    expected = []
    for pid, base, frame in ((10, 0x100000, 1), (10, 0x200000, 2), (11, 0x100000, 3)):
        layer = FakeProcLayer(b'\x00' * frame * PAGE + body.ljust(PAGE, b'\x00'), {})
        layer.map_virtual(base, [frame])
        spaces.append(SimpleNamespace(
            physical_mode=False, process_id=pid, source=ProcessLayerSource(layer),
            blocks=StreamReader(tmp_path / 'unused').iter_vad_blocks(layer, base, base + PAGE - 1),
            translate=lambda va, frame=frame: frame * PAGE + va % PAGE, skip_pages=frozenset(),
        ))
        expected.append({'process_id': pid, 'virtual_address': f'0x{base:016X}',
                         'physical_offset': f'0x{frame * PAGE:016X}'})
    images, [workflow], _ = _carve(SimpleNamespace(spaces=spaces, recovery=SimpleNamespace(physical=None, pfn=None)))
    assert images == []
    assert workflow.execution_parameters['source_locations'] == expected
    assert set(workflow.physical_offsets) == {location['physical_offset'] for location in expected}
    _export(tmp_path, [], [workflow])
    assert json.loads(json.dumps(workflow.to_dict()))['execution_parameters']['source_locations'] == expected
    assert (tmp_path / workflow.raw_file_path).read_bytes() == body


def test_virtual_workflow_bytes_match_the_physical_provenance(mock_memory_image, fake_ingestor, symbols_dir):
    memory = mock_memory_image("T1")
    ingestion = fake_ingestor(memory, SymbolManager(symbols_dir))
    try:
        worker = next(space for space in ingestion.spaces if space.process_id == 13484)
        _, records, _ = _carve(ingestion)
        [workflow] = [record for record in records if record.execution_parameters.get("prompt_id")]
        address = int(workflow.virtual_address, 16)
        offset = worker.translate(address)
        [location] = workflow.execution_parameters["source_locations"]
        assert workflow.physical_offsets == [location["physical_offset"]] == [f"0x{offset:016X}"]
        assert workflow.raw_bytes == memory.read_bytes()[offset:offset + len(workflow.raw_bytes)]
    finally:
        ingestion.close()


def test_f9_byte_distinct_and_different_encoding_copies_are_separate(tmp_path):
    graph = api_prompt()
    bodies = [json.dumps(graph).encode(), json.dumps(graph, separators=(',', ':')).encode(),
              json.dumps(graph).encode('utf-16-le'), prompt_request_body(graph, 'one'),
              prompt_request_body(graph, 'two')]
    carver = WorkflowCarver()
    records = []
    for i, body in enumerate(bodies):
        records += carver.carve_stream(body, base_physical_offset=i * 10000)
    assert len(records) == len(bodies)
    assert len({r.canonical_json_sha256 for r in records}) == 1
    _export(tmp_path, [], records)
    assert [(tmp_path / r.raw_file_path).read_bytes() for r in records] == bodies


def test_o1_integer_floats_normalize_recursively_and_png_links_are_strict():
    obj = {'nested': [8.0, {'denoise': 1.0, 'fraction': 1.25, 'yes': True, 'no': False}]}
    expected = '{"nested":[8,{"denoise":1,"fraction":1.25,"no":false,"yes":true}]}'
    assert canonical_json(obj) == expected
    assert KeyMatcher().canonicalize_prompt_json(obj) == expected
    graph = api_prompt()
    graph['3']['inputs'].update(cfg=8, denoise=1)
    [workflow] = WorkflowCarver().carve_stream(json.dumps(graph).encode())
    embedded = json.loads(json.dumps(graph))
    embedded['3']['inputs'].update(cfg=8.0, denoise=1.0)
    [image] = PngCarver().carve_stream(png_bytes(noise_image(64, 48), {'prompt': json.dumps(embedded)}))
    [link] = correlate([image], [workflow]).links
    assert link.link_class == 'STRICT' and 'CANONICAL_PROMPT_HASH' in link.matched_keys
    assert graph['3']['inputs']['cfg'] == 8 and type(obj['nested'][0]) is float


def test_f15_truncation_gaps_use_spec_reasons_and_positive_lengths():
    png = png_bytes(noise_image(20, 20))
    allowed = {'PAGE_NOT_PRESENT', 'TRANSLATION_FAULT', 'PAGED_OUT', 'CRC_FAILURE', 'NONCONTIGUOUS_PHYSICAL'}
    for cut in (33, 36, len(png) // 2, len(png) - 12):
        [record] = PngCarver().carve_stream(png[:cut])
        assert record.gaps
        assert all(g.length_bytes > 0 and g.reason in allowed for g in record.gaps)


def test_f17_palette_preview_records_dropped_transparency():
    image = Image.new('P', (4, 4), 0)
    image.putpalette([255, 0, 0] + [0] * 765)
    image.info['transparency'] = 0
    raw = io.BytesIO()
    image.save(raw, format='PNG')
    [record] = PngCarver().carve_stream(raw.getvalue()[:-12])
    assert record.preview_png
    assert record.derived_preview_metadata['transparency_dropped'] is True


def test_f18_zlib_error_preserves_rows_decoded_within_the_failing_block():
    pixels = b'\x33' * 10
    compressed = bytearray(zlib.compress(b'\x00\x33' * 10))
    compressed[-1] ^= 1
    meta = Ihdr(1, 10, 8, 0, 0).as_meta()
    assert PngCarver().decompress_partial_idat(bytes(compressed), meta) == (pixels, 10, 'ZLIB_DATA_ERROR')


def test_f19_standalone_workflows_keep_client_id():
    [workflow] = WorkflowCarver().carve_stream(prompt_request_body(api_prompt()))
    assert workflow.execution_parameters['client_id'] == '5f0c3c1e9a7b4e0f'


def _origin_stub():
    return SimpleNamespace(copy_origin=lambda pages, size: {'pages': sorted(pages), 'bytes': size})


def test_each_carved_copy_is_classified_by_the_pages_it_occupies(tmp_path):
    body = prompt_request_body(api_prompt())
    spaces = []
    for pid, base, frame in ((10, 0x100000, 5), (11, 0x100000, 9)):
        layer = FakeProcLayer(b'\x00' * frame * PAGE + body.ljust(PAGE, b'\x00'), {})
        layer.map_virtual(base, [frame])
        spaces.append(SimpleNamespace(
            physical_mode=False, process_id=pid, source=ProcessLayerSource(layer),
            blocks=StreamReader(tmp_path / 'unused').iter_vad_blocks(layer, base, base + PAGE - 1),
            translate=lambda va, frame=frame: frame * PAGE + va % PAGE, skip_pages=frozenset(),
        ))
    png = png_bytes(noise_image(20, 20, seed=7))
    # The same image twice in physical memory: once inside page 3, once across pages 6 and 7.
    raw = bytearray(10 * PAGE)
    raw[3 * PAGE + 16 : 3 * PAGE + 16 + len(png)] = png
    raw[7 * PAGE - 100 : 7 * PAGE - 100 + len(png)] = png
    assert len(png) > 100 and 3 * PAGE + 16 + len(png) < 4 * PAGE
    spaces.append(SimpleNamespace(physical_mode=True, process_id=None, source=None, translate=None,
                                  blocks=[SimpleNamespace(address=0, data=bytes(raw))], skip_pages=frozenset()))
    recovery = SimpleNamespace(physical=None, pfn=_origin_stub())
    [image], [workflow], _ = _carve(SimpleNamespace(spaces=spaces, recovery=recovery))
    assert workflow.page_origins == [{'pages': [5], 'bytes': len(workflow.raw_bytes)},
                                     {'pages': [9], 'bytes': len(workflow.raw_bytes)}]
    assert image.page_origins == [{'pages': [3], 'bytes': len(png)}, {'pages': [6, 7], 'bytes': len(png)}]
