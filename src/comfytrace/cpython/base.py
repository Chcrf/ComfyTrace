"""Byte-level interpretation of CPython objects for one interpreter build.

A ``CPythonLayout`` holds offsets only. Every method takes the process memory and the addresses it reads
explicitly, so one layout serves any number of processes and holds no state of its own. Offsets of object fields
are from the start of the object (``PyObject *``); offsets of dict key-table fields are from the start of the
``PyDictKeysObject``. 64-bit little-endian only.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from comfytrace.procmem import ProcessMemory, Unreadable

LayoutKey = tuple[str, tuple[int, int], str]
MINUS_ONE = np.uint64(0xFFFFFFFFFFFFFFFF)
WORD = 8


class CorruptObject(Unreadable):
    """The bytes are readable but do not form a valid object (stale or overwritten memory). ``address`` is where
    the invalid structure was read."""

    def __init__(self, message: str, address: int | None = None) -> None:
        super().__init__(message)
        self.address = address


def _words(offset: int, origin: int) -> int:
    distance, rest = divmod(offset - origin, WORD)
    if rest or distance <= 0:
        raise ValueError(f"field at {offset} does not follow {origin} by whole words")
    return distance


@dataclass(frozen=True)
class CPythonLayout:
    """Where one CPython build keeps the fields ComfyTrace reads, and how they are read."""

    keys: tuple[LayoutKey, ...]
    # PyObject / PyTypeObject
    object_header_size: int = 16
    refcnt_offset: int = 0
    type_offset: int = 8
    tp_name_offset: int = 24
    tp_basicsize_offset: int = 32
    tp_itemsize_offset: int = 40
    # str (PyASCIIObject / PyCompactUnicodeObject)
    str_length_offset: int = 16
    str_state_offset: int = 32
    str_ascii_data_offset: int = 40
    str_compact_data_offset: int = 56
    # int (PyLongObject, 3.12+ lv_tag) and float
    int_tag_offset: int = 16
    int_digits_offset: int = 24
    float_value_offset: int = 16
    # tuple and list
    tuple_size_offset: int = 16
    tuple_items_offset: int = 24
    list_size_offset: int = 16
    list_items_offset: int = 24
    list_allocated_offset: int = 32
    # dict and its key table
    dict_used_offset: int = 16
    dict_keys_offset: int = 32
    dict_values_offset: int = 40
    dk_header_offset: int = 8  # dk_log2_size, dk_log2_index_bytes, dk_kind: three consecutive bytes
    dk_nentries_offset: int = 24
    dk_indices_offset: int = 32
    unicode_entry_size: int = 16
    unicode_entry_key: int = 0
    unicode_entry_value: int = 8
    general_entry_size: int = 24
    general_entry_hash: int = 0
    general_entry_key: int = 8
    general_entry_value: int = 16
    split_values_offset: int = 0
    # set / frozenset
    set_fill_offset: int = 16
    set_used_offset: int = 24
    set_mask_offset: int = 32
    set_table_offset: int = 40
    set_entry_size: int = 16
    set_entry_key: int = 0  # setentry is {key, hash}: the hash is the other word of the 16-byte entry
    # bytes
    bytes_size_offset: int = 16
    bytes_hash_offset: int = 24
    bytes_data_offset: int = 32

    def __post_init__(self) -> None:
        if not self.keys:
            raise ValueError("a CPython layout must name at least one (image_os, version, build_flags) key")
        for entry, name in ((self.unicode_entry_size, "unicode_entry_size"),
                            (self.general_entry_size, "general_entry_size"), (self.set_entry_size, "set_entry_size")):
            if entry <= 0 or entry % WORD:
                raise ValueError(f"{name} must be a positive multiple of {WORD}")

    # -- object and type -----------------------------------------------------------------------------------------

    def refcount(self, mem: ProcessMemory, obj: int) -> int:
        return mem.q(obj + self.refcnt_offset)

    def type_address(self, mem: ProcessMemory, obj: int) -> int:
        return mem.q(obj + self.type_offset)

    def type_name(self, mem: ProcessMemory, type_addr: int) -> str:
        return mem.cstr(mem.q(type_addr + self.tp_name_offset))

    def is_none(self, mem: ProcessMemory, obj: int, none_type: int | None, image: Any) -> bool:
        """None is a header-only static singleton inside the runtime image. With the NoneType object known, the
        object's type is compared with it. Otherwise (its tp_name page may be paged out) None is recognised by
        shape: object and type both inside ``image``, tp_basicsize the bare header and tp_itemsize 0."""
        t = self.type_address(mem, obj)
        if none_type is not None:
            return t == none_type
        return (image is not None and image.contains(obj) and image.contains(t)
                and mem.q(t + self.tp_basicsize_offset) == self.object_header_size
                and mem.q(t + self.tp_itemsize_offset) == 0)

    # -- scalars ------------------------------------------------------------------------------------------------

    def string_length(self, mem: ProcessMemory, obj: int) -> int:
        return mem.q(obj + self.str_length_offset)

    def decode_str(self, mem: ProcessMemory, obj: int) -> str:
        n = self.string_length(mem, obj)
        state = struct.unpack("<I", mem.read(obj + self.str_state_offset, 4))[0]
        kind, compact, ascii_ = (state >> 2) & 7, state >> 5 & 1, state >> 6 & 1
        if not compact or kind not in (1, 2, 4) or n > 1 << 24:
            raise CorruptObject("str layout", obj)
        raw = mem.read(obj + (self.str_ascii_data_offset if ascii_ else self.str_compact_data_offset), n * kind)
        if kind == 1:
            return raw.decode("latin-1")
        return "".join(map(chr, struct.unpack(f"<{n}{'H' if kind == 2 else 'I'}", raw)))

    def decode_int(self, mem: ProcessMemory, obj: int) -> int:
        tag = mem.q(obj + self.int_tag_offset)
        ndigits, sign = tag >> 3, tag & 3
        if ndigits > 64 or sign == 3 or (sign == 1) != (ndigits == 0):
            raise CorruptObject("int tag", obj)
        digits = struct.unpack(f"<{ndigits}I", mem.read(obj + self.int_digits_offset, 4 * ndigits)) if ndigits else ()
        if digits and (digits[-1] == 0 or max(digits) >= 1 << 30):
            raise CorruptObject("int digits", obj)
        value = sum(d << (30 * i) for i, d in enumerate(digits))
        return -value if sign == 2 else value

    def decode_float(self, mem: ProcessMemory, obj: int) -> float:
        return float(struct.unpack("<d", mem.read(obj + self.float_value_offset, 8))[0])

    # -- sequences ----------------------------------------------------------------------------------------------

    def sequence_size(self, mem: ProcessMemory, obj: int, kind: str) -> int:
        """ob_size of a list (``kind == "list"``) or of anything with tuple storage (tuple, named tuple)."""
        return mem.q(obj + (self.list_size_offset if kind == "list" else self.tuple_size_offset))

    def sequence_base(self, mem: ProcessMemory, obj: int, kind: str) -> int:
        """Address of item 0: a list's ob_item vector, or the inline items of tuple storage."""
        return mem.q(obj + self.list_items_offset) if kind == "list" else obj + self.tuple_items_offset

    def sequence_item(self, mem: ProcessMemory, base: int, index: int) -> int:
        return mem.q(base + WORD * index)

    def list_allocated(self, mem: ProcessMemory, obj: int) -> int:
        return mem.q(obj + self.list_allocated_offset)

    # -- dict ---------------------------------------------------------------------------------------------------

    def _key_table(self, mem: ProcessMemory, keys: int) -> tuple[int, int, int, int]:
        log2_size, log2_index, dk_kind = mem.read(keys + self.dk_header_offset, 3)
        return log2_size, log2_index, dk_kind, mem.q(keys + self.dk_nentries_offset)

    def dict_items(self, mem: ProcessMemory, obj: int) -> list[tuple[int, int]]:
        """(key, value) addresses of every live entry, combined or split table, unicode or general keys."""
        q = mem.q
        used, keys, values = q(obj + self.dict_used_offset), q(obj + self.dict_keys_offset), \
            q(obj + self.dict_values_offset)
        log2_size, log2_index, dk_kind, nentries = self._key_table(mem, keys)
        if log2_size > 30 or nentries > 1 << 20:
            raise CorruptObject("dict keys", keys)
        index_width = 1 if log2_size < 8 else 2 if log2_size < 16 else 4 if log2_size < 32 else 8
        # dk_indices holds one index per slot; usable entries are at most 2/3 of the slots.
        # The shared empty key table has 0 slots but 8 index bytes, so only too few index bytes is corrupt.
        if (dk_kind not in (0, 1, 2) or (1 << log2_index) < (1 << log2_size) * index_width
                or nentries > ((1 << log2_size) * 2) // 3 or used > nentries):
            raise CorruptObject("dict key table inconsistent", keys)
        entries = keys + self.dk_indices_offset + (1 << log2_index)
        out = []
        if values:
            for i in range(nentries):
                entry = entries + self.unicode_entry_size * i
                key, value = q(entry + self.unicode_entry_key), q(values + self.split_values_offset + WORD * i)
                if key and value:
                    out.append((key, value))
        else:
            width, key_at, value_at = ((self.unicode_entry_size, self.unicode_entry_key, self.unicode_entry_value)
                                       if dk_kind else
                                       (self.general_entry_size, self.general_entry_key, self.general_entry_value))
            for i in range(nentries):
                key, value = q(entries + width * i + key_at), q(entries + width * i + value_at)
                if key and value:
                    out.append((key, value))
        if len(out) != used:
            raise CorruptObject(f"dict used {used} != {len(out)}", keys)
        return out

    def general_entries(self, mem: ProcessMemory, keys: int, limit: int) -> np.ndarray | None:
        """(hash, key, value) rows of a key table with general (DICT_KEYS_GENERAL) keys in one bounded read, or
        None for any other table, an implausible one or one that is not resident."""
        header = mem.try_read(keys + self.dk_header_offset, 3)
        if header is None or header[2] != 0 or header[0] > 30:
            return None
        nentries = mem.q(keys + self.dk_nentries_offset)
        if not 0 < nentries <= limit or nentries > ((1 << header[0]) * 2) // 3:
            return None
        raw = mem.try_read(keys + self.dk_indices_offset + (1 << header[1]), self.general_entry_size * nentries)
        if raw is None:
            return None
        rows = np.frombuffer(raw, np.uint64).reshape(-1, self.general_entry_size // WORD)
        return rows[:, [self.general_entry_hash // WORD, self.general_entry_key // WORD,
                        self.general_entry_value // WORD]]

    # -- set / frozenset ----------------------------------------------------------------------------------------

    def set_members(self, mem: ProcessMemory, obj: int, limit: int) -> list[int]:
        """Member addresses of a set or frozenset whose table has at most ``limit`` slots."""
        fill, used, mask, table = (mem.q(obj + offset) for offset in (
            self.set_fill_offset, self.set_used_offset, self.set_mask_offset, self.set_table_offset))
        if (mask + 1) & mask or not 7 <= mask < limit or not used <= fill <= mask + 1:
            raise CorruptObject("set layout", obj)
        rows = np.frombuffer(mem.read(table, self.set_entry_size * (mask + 1)), np.uint64)
        rows = rows.reshape(-1, self.set_entry_size // WORD)
        hash_at = (self.set_entry_key + WORD) % self.set_entry_size
        keys, hashes = rows[:, self.set_entry_key // WORD], rows[:, hash_at // WORD]
        # A discarded member leaves the dummy key with hash -1 (setobject.c); it counts in fill, not in used. No
        # live entry stores hash -1, since CPython never returns -1 as a hash.
        occupied = keys != 0
        live = occupied & (hashes != MINUS_ONE)
        members = [int(k) for k in keys[live]]
        if len(members) != used or int(occupied.sum()) != fill:
            raise CorruptObject(f"set used {used}/fill {fill} != {len(members)}/{int(occupied.sum())}", obj)
        return members

    # -- vectorized structural scans ----------------------------------------------------------------------------

    def iter_tuple_candidates(self, mem: ProcessMemory, tuple_type: int, lengths: Iterable[int]) -> Iterator[int]:
        """Addresses of objects whose type word is ``tuple_type`` and whose ob_size is one of ``lengths``."""
        gap = _words(self.tuple_size_offset, self.type_offset)
        t = np.uint64(tuple_type)
        sizes = [np.uint64(n) for n in sorted(set(lengths))]

        def test(w: np.ndarray) -> np.ndarray:
            size = w[gap:]
            wanted = size == sizes[0]
            for n in sizes[1:]:
                wanted |= size == n
            return np.nonzero((w[:-gap] == t) & wanted)[0]

        for addr, _w, _p in mem.scan_words(test):
            yield addr - self.type_offset

    def iter_dict_candidates(self, mem: ProcessMemory, dict_type: int) -> Iterator[tuple[int, int, int, int]]:
        """(object, ma_used, ma_keys, ma_values) of every object whose type word is ``dict_type``."""
        used_at, keys_at, values_at = (_words(o, self.type_offset) for o in (
            self.dict_used_offset, self.dict_keys_offset, self.dict_values_offset))
        tail = max(used_at, keys_at, values_at)
        t = np.uint64(dict_type)
        for addr, words, pos in mem.scan_words(lambda w: np.nonzero(w[:-tail] == t)[0]):
            yield (addr - self.type_offset, int(words[pos + used_at]), int(words[pos + keys_at]),
                   int(words[pos + values_at]))

    def iter_value_key_candidates(self, mem: ProcessMemory, wanted_values: np.ndarray) -> Iterator[tuple[int, int]]:
        """(key address, value) for every word equal to one of the sorted ``wanted_values`` read as the value of
        a unicode-keyed dict entry; the key is the word the entry layout puts before it."""
        back = self.unicode_entry_value - self.unicode_entry_key
        lo, hi = wanted_values[0], wanted_values[-1]

        def refers(w: np.ndarray) -> np.ndarray:
            pos = np.nonzero((w >= lo) & (w <= hi))[0]
            return pos[np.isin(w[pos], wanted_values)]

        for slot, words, pos in mem.scan_words(refers):
            try:
                key = mem.q(slot - back)
            except Unreadable:
                continue
            yield key, int(words[pos])

    def iter_bytes_candidates(self, mem: ProcessMemory, sizes: Iterable[int],
                              bytes_type: int | None) -> Iterator[tuple[int, int, int, str]]:
        """(object, ob_size, data address, state) of bytes objects whose size is in ``sizes``. A live one starts
        ``(refcnt, &PyBytes_Type, ob_size, ob_shash)``; a freed one keeps ``ob_size`` and ``ob_shash == -1`` after
        the allocator overwrites the header with free-list links, so the hash is the anchor and the type word
        only labels the state (LIVE or FREED_OR_UNTYPED)."""
        gap = _words(self.bytes_hash_offset, self.bytes_size_offset)
        want = np.array(sorted(set(sizes)), np.uint64)

        def test(w: np.ndarray) -> np.ndarray:
            # ob_shash == -1 is rare, an image size before it rarer: look the sizes up only where the hash matches.
            pos = np.nonzero(w[gap:] == MINUS_ONE)[0]
            return pos[np.isin(w[pos], want)]

        for addr, words, pos in mem.scan_words(test):
            obj = addr - self.bytes_size_offset
            try:
                type_word = self.type_address(mem, obj)
            except Unreadable:
                type_word = 0
            state = "LIVE" if bytes_type is not None and type_word == bytes_type else "FREED_OR_UNTYPED"
            yield obj, int(words[pos]), obj + self.bytes_data_offset, state
